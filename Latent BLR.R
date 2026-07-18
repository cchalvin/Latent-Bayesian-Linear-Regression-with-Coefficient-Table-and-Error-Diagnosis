# Required Libraries
if(!require(mvtnorm)) install.packages("mvtnorm"); library(mvtnorm)
if(!require(ggplot2)) install.packages("ggplot2"); library(ggplot2)
if(!require(gridExtra)) install.packages("gridExtra"); library(gridExtra)
if(!require(grid)) install.packages("grid"); library(grid)


BayesianLatentRegression <- function(formula, data, K = 2, 
                                     prior_lambda = 0.5, prior_alpha = 1.5, 
                                     max_iter = 100, tol = 1e-5) {

  
  # ---------------------------------------------------------
  # Step 1: Interface Parsing & Missing Data Initialization
  # ---------------------------------------------------------
  # 
  # 
  mf <- tryCatch({
    model.frame(formula, data = data, na.action = na.pass)
  }, error = function(e) stop("Formula parsing error. Ensure LHS uses cbind(). Details: ", e$message))
  
  Y <- model.response(mf)
  if(is.null(Y)) stop("No response variable found.")
  Y <- as.matrix(Y)
  X <- model.matrix(formula, mf)
  
  n <- nrow(Y); q <- ncol(Y); p <- ncol(X)
  
  # High-speed vectorized column mean imputation for X
  if(any(is.na(X))) {
    warning("NA detected in Predictors (X). Imputing with column means.")
    col_means_X <- colMeans(X, na.rm = TRUE)
    X[is.na(X)] <- rep(col_means_X, each = n)[is.na(X)]
  }
  
  # High-speed vectorized column mean imputation for Y
  na_mask <- is.na(Y)
  Y_imputed <- Y
  for(j in 1:q) {
    if(any(na_mask[, j])) {
      Y_imputed[na_mask[, j], j] <- mean(Y[!na_mask[, j], j])
    }
  }
  
  eps <- 1e-6
  Lambda_0 <- prior_lambda * diag(p)
  Lambda_eps <- Lambda_0 + diag(eps, p)
  
  # ---------------------------------------------------------
  # Part A: Null Model (Intercept only)
  # ---------------------------------------------------------
  X_null <- matrix(1, nrow = n, ncol = 1)
  XtX_null <- crossprod(X_null) + prior_lambda * diag(1) + diag(eps, 1)
  XtY_null <- crossprod(X_null, Y_imputed)
  B_null <- solve(XtX_null, XtY_null)  
  
  Resid_null <- Y_imputed - X_null %*% B_null
  Sigma_null <- (crossprod(Resid_null) + diag(eps, q)) / n
  
  LL_null <- sum(dmvnorm(Resid_null, sigma = Sigma_null, log = TRUE))
  df_null <- q + (q*(q+1))/2
  BIC_null <- -2 * LL_null + df_null * log(n)
  
  # ---------------------------------------------------------
  # Part B: Standard BLR (K = 1)
  # ---------------------------------------------------------
  XtX_std <- crossprod(X) + Lambda_eps
  XtY_std <- crossprod(X, Y_imputed)
  B_std <- solve(XtX_std, XtY_std)
  
  Resid_std <- Y_imputed - X %*% B_std
  Sigma_std <- (crossprod(Resid_std) + diag(eps, q)) / n
  
  LL_std <- sum(dmvnorm(Resid_std, sigma = Sigma_std, log = TRUE))
  df_std <- p*q + (q*(q+1))/2
  BIC_std <- -2 * LL_std + df_std * log(n)
  
  BF_null_std <- exp(-0.5 * (BIC_std - BIC_null))
  
  Cov_B_std <- kronecker(Sigma_std, solve(XtX_std))
  SE_B_std <- matrix(sqrt(diag(Cov_B_std)), nrow=p, ncol=q)
  
  # ---------------------------------------------------------
  # Part C: Latent BLR (K > 1) via Fast Vectorized EM
  # ---------------------------------------------------------
  set.seed(42)
  if(K > 1) {
    km <- kmeans(Y_imputed, centers = K, nstart = 10)
    gamma_ik <- matrix(0, n, K)
    for(k in 1:K) gamma_ik[km$cluster == k, k] <- 1
  } else {
    gamma_ik <- matrix(1, n, 1)
  }
  
  pi_k <- colMeans(gamma_ik)
  B_list <- replicate(K, B_std, simplify = FALSE)
  Sigma_list <- replicate(K, Sigma_std, simplify = FALSE)
  
  LL_history <- c()
  
  for(iter in 1:max_iter) {
    # --- M-Step ---
    for(k in 1:K) {
      w_k <- gamma_ik[, k]
      N_k <- sum(w_k)
      pi_k[k] <- (N_k + prior_alpha - 1) / (n + K*(prior_alpha - 1))
      
      # Vectorized scaling updates to replace costly matrix transformations
      XtWX <- crossprod(X, X * w_k) + Lambda_eps
      XtWY <- crossprod(X, Y_imputed * w_k)
      B_list[[k]] <- solve(XtWX, XtWY)
      
      Y_pred <- X %*% B_list[[k]]
      Y_imputed[na_mask] <- Y_pred[na_mask]
      
      Resid_k <- Y_imputed - X %*% B_list[[k]]
      Sigma_list[[k]] <- (crossprod(Resid_k, Resid_k * w_k) + diag(eps, q)) / (N_k + q + 1)
    }
    pi_k <- pi_k / sum(pi_k)
    
    # --- E-Step ---
    log_gamma <- matrix(0, n, K)
    for(k in 1:K) {
      Resid_k <- Y_imputed - X %*% B_list[[k]]
      log_gamma[, k] <- log(pi_k[k] + 1e-15) + dmvnorm(Resid_k, sigma = Sigma_list[[k]], log = TRUE)
    }
    
    max_log <- apply(log_gamma, 1, max)
    gamma_ik <- exp(log_gamma - max_log)
    gamma_ik <- gamma_ik / rowSums(gamma_ik)
    
    LL_latent <- sum(max_log + log(rowSums(exp(log_gamma - max_log))))
    LL_history <- c(LL_history, LL_latent)
    
    if(iter > 1 && abs(LL_history[iter] - LL_history[iter-1]) < tol) break
  }
  
  df_latent <- K*(p*q + (q*(q+1))/2) + (K-1)
  BIC_latent <- -2 * LL_latent + df_latent * log(n)
  BF_std_lat <- exp(-0.5 * (BIC_latent - BIC_std))
  
  # ---------------------------------------------------------
  # Part D: Tidying Coefficient Tables (With Unified Class Labels)
  # ---------------------------------------------------------
  get_bf_stars <- function(mean_val, sd_val) {
    if (is.na(mean_val) || is.na(sd_val) || sd_val == 0) return("")
    z_score <- mean_val / sd_val
    bf <- exp(0.5 * (z_score^2)) # Wald-based localized Bayesian evidence factor
    if (bf > 100) return(" ***")
    if (bf > 50)  return(" **")
    if (bf > 10)  return(" *")
    if (bf > 5)   return(" .")
    return("")
  }
  
  make_spacer_row <- function() {
    data.frame(Model="", Class="", Class_Proportion=NA,
               Output="", Variable="", Mean=NA, SD=NA, 
               Estimate_CI="", stringsAsFactors=FALSE)
  }
  
  col_Y <- colnames(Y_imputed); if(is.null(col_Y)) col_Y <- paste0("Y", 1:q)
  col_X <- colnames(X)
  
  results_list <- list()
  
  # --- 1. Standard BLR Preallocated Block ---
  n_std_rows <- q * (p + 1)
  std_Model <- rep("Standard BLR", n_std_rows)
  std_Class <- rep("1 (1.00)", n_std_rows)
  std_Class_Proportion <- rep(1.0, n_std_rows)
  std_Output <- rep("", n_std_rows)
  std_Variable <- rep("", n_std_rows)
  std_Mean <- rep(0, n_std_rows)
  std_SD <- rep(0, n_std_rows)
  std_Estimate_CI <- rep("", n_std_rows)
  
  idx <- 1
  for(j in 1:q) {
    # Standard Error Row
    val_mean_se <- sqrt(Sigma_std[j, j])
    val_sd_se <- val_mean_se / sqrt(2 * n)
    std_Output[idx] <- col_Y[j]
    std_Variable[idx] <- "Standard Error"
    std_Mean[idx] <- val_mean_se
    std_SD[idx] <- val_sd_se
    std_Estimate_CI[idx] <- sprintf("%.3f (%.3f, %.3f)%s", val_mean_se, val_mean_se - 1.96*val_sd_se, val_mean_se + 1.96*val_sd_se, get_bf_stars(val_mean_se, val_sd_se))
    idx <- idx + 1
    
    # Predictor Rows
    for(i in 1:p) {
      val_B <- B_std[i, j]
      val_SE <- SE_B_std[i, j]
      std_Output[idx] <- col_Y[j]
      std_Variable[idx] <- col_X[i]
      std_Mean[idx] <- val_B
      std_SD[idx] <- val_SE
      std_Estimate_CI[idx] <- sprintf("%.3f (%.3f, %.3f)%s", val_B, val_B - 1.96*val_SE, val_B + 1.96*val_SE, get_bf_stars(val_B, val_SE))
      idx <- idx + 1
    }
  }
  results_list[[1]] <- data.frame(
    Model = std_Model, Class = std_Class, Class_Proportion = std_Class_Proportion,
    Output = std_Output, Variable = std_Variable, Mean = std_Mean, SD = std_SD, 
    Estimate_CI = std_Estimate_CI, stringsAsFactors = FALSE
  )
  results_list[[2]] <- make_spacer_row()
  
  # --- 2. Latent BLR Preallocated Block ---
  SE_list <- list() 
  latent_df_list <- list()
  
  if(K > 1) {
    for(k in 1:K) {
      XtWX <- crossprod(X, X * gamma_ik[, k]) + Lambda_eps
      Cov_B_k <- tryCatch({
        kronecker(Sigma_list[[k]], solve(XtWX))
      }, error = function(e) {
        kronecker(Sigma_list[[k]], solve(XtWX + diag(1e-5, p)))
      })
      
      SE_B_k <- matrix(sqrt(pmax(0, diag(Cov_B_k))), nrow=p, ncol=q)
      SE_list[[k]] <- SE_B_k
      
      n_latent_rows <- q * (p + 1)
      class_combined_label <- sprintf("%d (%.2f)", k, pi_k[k])
      
      lat_Model <- rep("Latent BLR", n_latent_rows)
      lat_Class <- rep(class_combined_label, n_latent_rows)
      lat_Class_Proportion <- rep(pi_k[k], n_latent_rows)
      lat_Output <- rep("", n_latent_rows)
      lat_Variable <- rep("", n_latent_rows)
      lat_Mean <- rep(0, n_latent_rows)
      lat_SD <- rep(0, n_latent_rows)
      lat_Estimate_CI <- rep("", n_latent_rows)
      
      idx <- 1
      N_k <- sum(gamma_ik[, k])
      for(j in 1:q) {
        # Standard Error Row
        val_mean_se <- sqrt(Sigma_list[[k]][j, j])
        val_sd_se <- val_mean_se / sqrt(2 * N_k)
        lat_Output[idx] <- col_Y[j]
        lat_Variable[idx] <- "Standard Error"
        lat_Mean[idx] <- val_mean_se
        lat_SD[idx] <- val_sd_se
        lat_Estimate_CI[idx] <- sprintf("%.3f (%.3f, %.3f)%s", val_mean_se, val_mean_se - 1.96*val_sd_se, val_mean_se + 1.96*val_sd_se, get_bf_stars(val_mean_se, val_sd_se))
        idx <- idx + 1
        
        # Predictor Rows
        for(i in 1:p) {
          val_B  <- B_list[[k]][i, j]
          val_SE <- SE_B_k[i, j]
          lat_Output[idx] <- col_Y[j]
          lat_Variable[idx] <- col_X[i]
          lat_Mean[idx] <- val_B
          lat_SD[idx] <- val_SE
          lat_Estimate_CI[idx] <- sprintf("%.3f (%.3f, %.3f)%s", val_B, val_B - 1.96*val_SE, val_B + 1.96*val_SE, get_bf_stars(val_B, val_SE))
          idx <- idx + 1
        }
      }
      latent_df_list[[k]] <- data.frame(
        Model = lat_Model, Class = lat_Class, Class_Proportion = lat_Class_Proportion,
        Output = lat_Output, Variable = lat_Variable, Mean = lat_Mean, SD = lat_SD, 
        Estimate_CI = lat_Estimate_CI, stringsAsFactors = FALSE
      )
    }
    
    for(k in 1:K) {
      results_list[[length(results_list) + 1]] <- latent_df_list[[k]]
      if(k < K) results_list[[length(results_list) + 1]] <- make_spacer_row()
    }
  }
  
  tidy_results <- do.call(rbind, results_list)
  
  # ---------------------------------------------------------
  # Part E: Error Table (Prior vs Posterior Optimization)
  # ---------------------------------------------------------
  make_error_spacer_row <- function() {
    data.frame(Class="", Output="", Variable="", Prior_Mean=NA, Post_Mean=NA,
               Mean_Diff=NA, `Mean_change%`=NA, Prior_SD=NA, Post_SD=NA, SD_Ratio=NA,
               Post_95_CI="", check.names=FALSE, stringsAsFactors=FALSE)
  }
  
  error_df_list <- list()
  if(K > 1) {
    for(k in 1:K) {
      SE_B_k <- SE_list[[k]]
      n_err_rows <- p * q
      class_prop_label <- sprintf("%d (%.2f)", k, pi_k[k])
      
      err_Class <- rep(class_prop_label, n_err_rows)
      err_Output <- rep("", n_err_rows)
      err_Variable <- rep("", n_err_rows)
      err_Prior_Mean <- rep(0, n_err_rows)
      err_Post_Mean <- rep(0, n_err_rows)
      err_Mean_Diff <- rep(0, n_err_rows)
      err_Mean_ChangePct <- rep("", n_err_rows) # Character vector to handle "Inf" strings cleanly
      err_Prior_SD <- rep(0, n_err_rows)
      err_Post_SD <- rep(0, n_err_rows)
      err_SD_Ratio <- rep(0, n_err_rows)
      err_Post_95_CI <- rep("", n_err_rows)
      
      idx <- 1
      for(j in 1:q) {
        prior_sd_j <- sqrt(Sigma_null[j, j]) / sqrt(prior_lambda)
        for(i in 1:p) {
          prior_m  <- 0.0  
          post_m   <- B_list[[k]][i, j]
          post_sd  <- SE_B_k[i, j]
          
          m_diff <- post_m - prior_m
          
          # Implementation of the structural Mean Percentage Change formula
          if (post_m == 0) {
            m_change_str <- "Inf"
          } else {
            m_change_str <- sprintf("%.2f%%", ((post_m - prior_m) / post_m) * 100)
          }
          
          sd_ratio <- post_sd / prior_sd_j
          low <- post_m - 1.96 * post_sd
          up  <- post_m + 1.96 * post_sd
          
          err_Output[idx] <- col_Y[j]
          err_Variable[idx] <- col_X[i]
          err_Prior_Mean[idx] <- prior_m
          err_Post_Mean[idx] <- post_m
          err_Mean_Diff[idx] <- m_diff
          err_Mean_ChangePct[idx] <- m_change_str
          err_Prior_SD[idx] <- prior_sd_j
          err_Post_SD[idx] <- post_sd
          err_SD_Ratio[idx] <- sd_ratio
          err_Post_95_CI[idx] <- sprintf("(%.3f, %.3f)", low, up)
          
          idx <- idx + 1
        }
      }
      error_df_list[[k]] <- data.frame(
        Class = err_Class, Output = err_Output, Variable = err_Variable,
        Prior_Mean = err_Prior_Mean, Post_Mean = err_Post_Mean, Mean_Diff = err_Mean_Diff,
        `Mean_change%` = err_Mean_ChangePct, Prior_SD = err_Prior_SD, Post_SD = err_Post_SD,
        SD_Ratio = err_SD_Ratio, Post_95_CI = err_Post_95_CI,
        check.names = FALSE, stringsAsFactors = FALSE
      )
    }
  }
  
  error_list <- list()
  if(K > 1) {
    for(k in 1:K) {
      error_list[[length(error_list) + 1]] <- error_df_list[[k]]
      if(k < K) error_list[[length(error_list) + 1]] <- make_error_spacer_row()
    }
  }
  error_comparison_table <- do.call(rbind, error_list)
  
  # ---------------------------------------------------------
  # Part F: Visualizations (q Isolated Full-Grid Graph Modules)
  # ---------------------------------------------------------
  Y_mat <- as.matrix(Y_imputed)
  X_mat <- as.matrix(X)
  cluster_assignment <- factor(apply(gamma_ik, 1, which.max), levels = 1:K)
  
  non_intercept_idx <- which(apply(X_mat, 2, function(x) length(unique(x)) > 1))
  col_X_plot <- colnames(X_mat); if(is.null(col_X_plot)) col_X_plot <- paste0("X", 1:p)
  col_Y_plot <- colnames(Y_mat); if(is.null(col_Y_plot)) col_Y_plot <- paste0("Y", 1:q)
  
  beta_ols <- solve(t(X_mat) %*% X_mat + 1e-8 * diag(p)) %*% (t(X_mat) %*% Y_mat)
  X_mean <- colMeans(X_mat)
  
  composite_plots_list <- list()
  
  for(d in 1:q) {
    y_name <- col_Y_plot[d]
    y_vals <- Y_mat[, d]
    sub_plots_for_y <- list()
    
    for(j_idx in seq_along(non_intercept_idx)) {
      j <- non_intercept_idx[j_idx]
      x_vals <- X_mat[, j]
      x_name <- col_X_plot[j]
      
      is_categorical <- is.factor(x_vals) || is.character(x_vals) || length(unique(x_vals)) <= 10
      x_grid <- if(is_categorical) sort(unique(x_vals)) else seq(min(x_vals, na.rm=TRUE), max(x_vals, na.rm=TRUE), length.out=200)
      
      X_pred_ols <- matrix(rep(X_mean, each = length(x_grid)), nrow = length(x_grid))
      X_pred_ols[, j] <- x_grid
      y_pred_ols <- X_pred_ols %*% beta_ols[, d]
      df_ols <- data.frame(X = x_grid, Y = y_pred_ols)
      
      pred_list <- list()
      for(k in 1:K) {
        beta_k <- B_list[[k]][, d]
        sigma_kd <- sqrt(Sigma_list[[k]][d, d])
        
        X_pred_k <- matrix(rep(X_mean, each = length(x_grid)), nrow = length(x_grid))
        X_pred_k[, j] <- x_grid
        y_pred_k <- X_pred_k %*% beta_k
        
        pred_list[[k]] <- data.frame(
          X = x_grid, Y = y_pred_k, Cluster = factor(k, levels = 1:K),
          ymin = y_pred_k - 1.96 * sigma_kd, ymax = y_pred_k + 1.96 * sigma_kd
        )
      }
      df_preds <- do.call(rbind, pred_list)
      
      if(is_categorical) {
        df_points <- data.frame(X = factor(x_vals), Y = y_vals, Cluster = cluster_assignment)
        df_preds$X <- factor(df_preds$X, levels = levels(df_points$X))
        df_ols$X <- factor(df_ols$X, levels = levels(df_points$X))
        
        p_sub <- ggplot() +
          geom_boxplot(data = df_points, aes(x = X, y = Y, fill = Cluster), position = position_dodge(width = 0.8), width = 0.7, alpha = 0.4, outlier.size = 1) +
          geom_line(data = df_preds, aes(x = as.numeric(X), y = Y, color = Cluster), linewidth = 1.2) +
          geom_line(data = df_ols, aes(x = as.numeric(X), y = Y), linetype = "dashed", color = "black", linewidth = 0.8) +
          theme_minimal(base_size = 10) + labs(title = paste(y_name, "vs", x_name), x = x_name, y = y_name) +
          scale_color_brewer(palette = "Set1") + scale_fill_brewer(palette = "Set1")
      } else {
        df_points <- data.frame(X = x_vals, Y = y_vals, Cluster = cluster_assignment)
        
        p_sub <- ggplot() +
          geom_point(data = df_points, aes(x = X, y = Y, color = Cluster), alpha = 0.4, size = 1.2) +
          geom_ribbon(data = df_preds, aes(x = X, ymin = ymin, ymax = ymax, fill = Cluster), alpha = 0.1) +
          geom_line(data = df_preds, aes(x = X, y = Y, color = Cluster), linewidth = 1.2) +
          geom_line(data = df_ols, aes(x = X, y = Y), linetype = "dashed", color = "black", linewidth = 0.8) +
          theme_minimal(base_size = 10) + labs(title = paste(y_name, "vs", x_name), x = x_name, y = y_name) +
          scale_color_brewer(palette = "Set1") + scale_fill_brewer(palette = "Set1")
      }
      sub_plots_for_y[[length(sub_plots_for_y) + 1]] <- p_sub
    }
    
    if(length(sub_plots_for_y) > 0) {
      n_cols <- min(length(sub_plots_for_y), 2)
      composite_plot <- gridExtra::arrangeGrob(
        grobs = sub_plots_for_y, ncol = n_cols,
        top = grid::textGrob(paste("Diagnostic Graph Panel for Response Column:", y_name), gp = grid::gpar(fontsize = 13, fontface = "bold"))
      )
      grid::grid.newpage()
      grid::grid.draw(composite_plot)
      composite_plots_list[[y_name]] <- composite_plot
    }
  }
  
  metrics <- data.frame(
    Comparison = c("Null vs Std", "Std vs Latent"),
    BIC_Diff = c(BIC_null - BIC_std, BIC_std - BIC_latent),
    BayesFactor = c(1/BF_null_std, 1/BF_std_lat)
  )
  metrics$Better_Model <- ifelse(metrics$BIC_Diff > 0, sub(".* vs ", "", metrics$Comparison), sub(" vs .*", "", metrics$Comparison))
  
  # ---------------------------------------------------------
  # Part G: Unified Execution Output Reporter
  # ---------------------------------------------------------
  cat("\n", paste(rep("=", 95), collapse = ""), "\n")
  cat("                    BAYESIAN LATENT REGRESSION METRIC SUMMARY REPORT\n")
  cat(paste(rep("=", 95), collapse = ""), "\n\n")
  
  cat("[1] EXPECTATION-MAXIMIZATION MODEL CONVERGENCE PROFILE\n")
  cat("    Evaluated Iterations To Completion :", length(LL_history), "\n")
  cat("    Terminal Combined Log-Likelihood   :", sprintf("%.5f", tail(LL_history, 1)), "\n\n")
  
  cat("[2] MODEL SELECTION MATRIX (BAYESIAN INFORMATION CRITERION)\n")
  print(metrics, row.names = FALSE)
  cat("\n")
  
  cat("[3] POSTERIOR ESTIMATES & REGRESSION PARAMETERS TABLE\n")
  print(tidy_results[, c("Model", "Class", "Output", "Variable", "Mean", "SD", "Estimate_CI")], row.names = FALSE, na.print = "")
  cat("\n")
  
  if(K > 1) {
    cat("[4] CRITICAL COVARIATE CONTRACTION SUMMARY: PRIOR LATENT VS POSTERIOR LATENT\n")
    print(error_comparison_table, row.names = FALSE, na.print = "")
    cat("\n")
  }
  cat(paste(rep("=", 95), collapse = ""), "\n\n")
  
  return(list(
    Y_Imputed = Y_imputed,
    Coefficients = tidy_results,
    Error_Comparison = error_comparison_table,
    Model_Metrics = metrics,
    Plots = composite_plots_list,
    EM_Convergence = LL_history
  ))
}


set.seed(123)
N_nodes <- 2000; m <- 3; N <- N_nodes * m
node_ids <- rep(1:N_nodes, each = m)

# Continuous Predictor
X_cont <- seq(-3, 3, length.out=N) + rnorm(N, sd=0.5)
# Categorical/Factor Predictor
X_cat <- sample(1:3, N, replace = TRUE)

Y_val1 <- numeric(N)
Y_val2 <- numeric(N)
true_cls <- rep(sample(0:1, N_nodes, replace=TRUE), each = m)

# Building the Target Y with both variables affecting it
Y_val1[true_cls == 0] <- 2.0 + 3.0 ^ X_cont[true_cls == 0] + 1.5 * X_cat[true_cls == 0] + rnorm(sum(true_cls==0))
Y_val1[true_cls == 1] <- -1.0 - 2.0 ^ X_cont[true_cls == 1] - 2.5 * X_cat[true_cls == 1] + rnorm(sum(true_cls==1))


# Building the Target Y with both variables affecting it
Y_val2[true_cls == 0] <- 12.0 + 0.5 ^ X_cont[true_cls == 0] + 15 * X_cat[true_cls == 0] + rnorm(sum(true_cls==0))
Y_val2[true_cls == 1] <- -11.0 - 2.5 ^ X_cont[true_cls == 1] - 2.5 * X_cat[true_cls == 1] + rnorm(sum(true_cls==1))


df <- data.frame(response1 = Y_val1, response2 = Y_val2, predictor_cont = X_cont, predictor_cat = X_cat, group = node_ids)


# Execute the model with a single 'data' argument
results <- BayesianLatentRegression(
  formula = cbind(response1, response2) ~ predictor_cont + predictor_cat, 
  data = df, 
  K = 4
)








