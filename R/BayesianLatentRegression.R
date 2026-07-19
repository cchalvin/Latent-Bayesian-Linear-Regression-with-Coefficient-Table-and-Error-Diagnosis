#' @title Bayesian Latent Generalized Linear Regression
#'
#' @description Fits a finite mixture of Bayesian linear or generalized linear regression 
#' models to multivariate response data using a Quasi-Newton Expectation-Maximization 
#' (EM) algorithm. Accommodates missing response data via iterative conditional mean imputation 
#' and outputs variance contraction profiles comparing prior distributions against 
#' empirical class-level posterior estimates.
#'
#' @param formula An object of class \code{\link[stats]{formula}}. The left-hand side must 
#' contain the multivariate response variables wrapped in \code{cbind()}, e.g., \code{cbind(Y1, Y2) ~ X1 + X2}.
#' @param data A data frame containing the variables defined in \code{formula}.
#' @param K Integer. The number of latent classes to extract. Default is \code{2}.
#' @param prior_lambda Numeric. Ridge-like precision hyperparameter ($\lambda$) for the Gaussian 
#' prior on regression coefficients. Default is \code{0.5}.
#' @param prior_alpha Numeric. Dirichlet concentration parameter ($\alpha$) acting as a symmetric 
#' prior over latent class proportions. Default is \code{1.5}.
#' @param max_iter Integer. Maximum number of iterations permitted for the EM algorithm. Default is \code{100}.
#' @param tol Numeric. Convergence tolerance criterion based on the absolute difference in joint 
#' log-likelihood between sequential steps. Default is \code{1e-5}.
#' @param family Character string or function indicating the error distribution and link function. 
#' Supported options match standard GLM string entries: \code{"gaussian"}, \code{"log"}, or \code{"inverse"}.
#'
#' @details 
#' The model assumes observations originate from $K$ hidden subgroups. For a given class $k$, 
#' the parameters follow a conditional Gaussian prior setup. The EM loop optimizes parameters 
#' via Quasi-Newton optimization (\code{"BFGS"}) within the M-step. 
#' 
#' The returned \code{Error_Comparison} profiling table treats the residual standard deviation 
#' (\code{"Residual SD"}) explicitly as a latent parameter variable, calculating its shift from 
#' the global null variance baseline to the posterior class-specific state.
#'
#' @return A structured \code{list} containing:
#' \item{Y_Imputed}{A matrix of responses where missing entries are replaced by their conditional expected values.}
#' \item{Coefficients}{A data frame mapping posterior parameter averages, standard errors, and 95\% credible intervals.}
#' \item{Error_Comparison}{A variance contraction profiling matrix tracking \code{Prior_Mean}, \code{Post_Mean}, and \code{SD_Ratio} metrics.}
#' \item{Model_Metrics}{Model selection table displaying log-likelihood limits, BIC configurations, and Bayes Factors.}
#' \item{Plots}{A list of structured grid panels visualizing class trajectories across continuous or categorical fields.}
#' \item{EM_Convergence}{Numeric vector tracing the log-likelihood realization trajectory.}
#'
#' @importFrom mvtnorm dmvnorm
#' @importFrom ggplot2 ggplot geom_point geom_ribbon geom_line geom_boxplot theme_minimal labs scale_color_brewer scale_fill_brewer aes position_dodge
#' @importFrom gridExtra arrangeGrob
#' @importFrom grid textGrob gpar grid.newpage grid.draw
#' @export
BayesianLatentRegression <- function(formula, data, K = 2, 
                                     prior_lambda = 0.5, prior_alpha = 1.5, 
                                     max_iter = 100, tol = 1e-5,
                                     family = "gaussian") {
  
  if (is.character(family)) {
    family_obj <- tryCatch(get(family)(), error = function(e) stop("Invalid family string provided."))
  } else if (is.function(family)) {
    family_obj <- family()
  } else {
    family_obj <- family
  }
  
  if (!inherits(family_obj, "family") || is.null(family_obj$linkinv)) {
    stop("Input 'family' must be a valid R family object (e.g., binomial(), gaussian(), poisson(), etc.)")
  }
  
  g_func <- function(X, B) {
    eta <- X %*% B
    mu <- family_obj$linkinv(eta)
    if (any(!is.finite(mu))) {
      mu[is.na(mu) | !is.finite(mu)] <- .Machine$double.eps
    }
    return(mu)
  }  
  
  # ---------------------------------------------------------
  # Step 1: Interface Parsing & Missing Data Initialization
  # ---------------------------------------------------------
  mf <- tryCatch({
    model.frame(formula, data = data, na.action = na.pass)
  }, error = function(e) stop("Formula parsing error. Ensure LHS uses cbind(). Details: ", e$message))
  
  Y <- model.response(mf)
  if(is.null(Y)) stop("No response variable found.")
  Y <- as.matrix(Y)
  X <- model.matrix(formula, mf)
  
  n <- nrow(Y); q <- ncol(Y); p <- ncol(X)
  
  if(any(is.na(X))) {
    warning("NA detected in Predictors (X). Imputing with column means.")
    col_means_X <- colMeans(X, na.rm = TRUE)
    X[is.na(X)] <- rep(col_means_X, each = n)[is.na(X)]
  }
  
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
  # Part A: Null Model (Linear Intercept Only Baseline)
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
  # Part B: Standard Robust Generalized BLR (K = 1)
  # ---------------------------------------------------------
  if (family == "gaussian") {
    B_std_init <- solve(crossprod(X) + Lambda_eps, crossprod(X, Y_imputed))
  } else if (family == "log") {
    Y_link <- log(abs(Y_imputed) + 1e-3)
    B_std_init <- solve(crossprod(X) + Lambda_eps, crossprod(X, Y_link))
  } else if (family == "inverse") {
    Y_link <- 1 / (Y_imputed + sign(Y_imputed) * 1e-3)
    B_std_init <- solve(crossprod(X) + Lambda_eps, crossprod(X, Y_link))
  }
  
  g_check <- tryCatch(g_func(X, B_std_init), error = function(e) NULL)
  if(is.null(g_check) || any(!is.finite(g_check))) {
    B_std_init <- matrix(0.1, nrow = p, ncol = q)
  }
  
  Sigma_std_init <- (crossprod(Y_imputed - g_func(X, B_std_init)) + diag(eps, q)) / n
  
  obj_std <- function(B_vec, X, Y, Sigma, Lambda_eps, q, p) {
    B_mat <- matrix(B_vec, nrow = p, ncol = q)
    g_val <- g_func(X, B_mat)
    if(any(!is.finite(g_val))) return(1e20) 
    
    Resid <- Y - g_val
    Sigma_inv <- tryCatch(solve(Sigma), error = function(e) solve(Sigma + diag(1e-4, ncol(Sigma))))
    quad_form <- rowSums((Resid %*% Sigma_inv) * Resid)
    
    val <- 0.5 * sum(quad_form) + 0.5 * sum(diag(t(B_mat) %*% Lambda_eps %*% B_mat))
    if(!is.finite(val)) return(1e20)
    return(val)
  }
  
  opt_std <- optim(par = as.vector(B_std_init), fn = obj_std, X = X, Y = Y_imputed,
                   Sigma = Sigma_std_init, Lambda_eps = Lambda_eps, q = q, p = p, 
                   method = "BFGS", hessian = TRUE)
  
  B_std <- matrix(opt_std$par, nrow = p, ncol = q)
  Resid_std <- Y_imputed - g_func(X, B_std)
  Sigma_std <- (crossprod(Resid_std) + diag(eps, q)) / n
  
  LL_std <- sum(dmvnorm(Resid_std, sigma = Sigma_std, log = TRUE))
  df_std <- p*q + (q*(q+1))/2
  BIC_std <- -2 * LL_std + df_std * log(n)
  
  BF_null_std <- exp(-0.5 * (BIC_std - BIC_null))
  
  Cov_B_std <- tryCatch(solve(opt_std$hessian), error = function(e) solve(opt_std$hessian + diag(1e-5, p*q)))
  SE_B_std <- matrix(sqrt(pmax(0, diag(Cov_B_std))), nrow = p, ncol = q)
  
  # ---------------------------------------------------------
  # Part C: Latent Generalized BLR (K > 1) via Quasi-Newton EM
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
  
  obj_k <- function(B_vec, X, Y, w_k, Sigma, Lambda_eps, q, p) {
    B_mat <- matrix(B_vec, nrow = p, ncol = q)
    g_val <- g_func(X, B_mat)
    if(any(!is.finite(g_val))) return(1e20)
    
    Resid <- Y - g_val
    Sigma_inv <- tryCatch(solve(Sigma), error = function(e) solve(Sigma + diag(1e-4, ncol(Sigma))))
    quad_form <- rowSums((Resid %*% Sigma_inv) * Resid)
    
    val <- 0.5 * sum(w_k * quad_form) + 0.5 * sum(diag(t(B_mat) %*% Lambda_eps %*% B_mat))
    if(!is.finite(val)) return(1e20)
    return(val)
  }
  
  for(iter in 1:max_iter) {
    for(k in 1:K) {
      w_k <- gamma_ik[, k]
      N_k <- sum(w_k)
      pi_k[k] <- (N_k + prior_alpha - 1) / (n + K*(prior_alpha - 1))
      
      opt_k <- optim(par = as.vector(B_list[[k]]), fn = obj_k, X = X, Y = Y_imputed,
                     w_k = w_k, Sigma = Sigma_list[[k]], Lambda_eps = Lambda_eps,
                     q = q, p = p, method = "BFGS", hessian = FALSE)
      
      B_list[[k]] <- matrix(opt_k$par, nrow = p, ncol = q)
      
      Y_pred <- g_func(X, B_list[[k]])
      Y_imputed[na_mask] <- Y_pred[na_mask]
      
      Resid_k <- Y_imputed - Y_pred
      Sigma_list[[k]] <- (crossprod(Resid_k, Resid_k * w_k) + diag(eps, q)) / (N_k + q + 1)
    }
    pi_k <- pi_k / sum(pi_k)
    
    log_gamma <- matrix(0, n, K)
    for(k in 1:K) {
      Resid_k <- Y_imputed - g_func(X, B_list[[k]])
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
  # Part D: Tidying Coefficient Tables (Hessian Inversion)
  # ---------------------------------------------------------
  get_bf_stars <- function(mean_val, sd_val) {
    if (is.na(mean_val) || is.na(sd_val) || sd_val == 0) return("")
    z_score <- mean_val / sd_val
    bf <- exp(0.5 * (z_score^2)) 
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
    val_mean_se <- sqrt(Sigma_std[j, j])
    val_sd_se <- val_mean_se / sqrt(2 * n)
    std_Output[idx] <- col_Y[j]
    std_Variable[idx] <- "Standard Error"
    std_Mean[idx] <- val_mean_se
    std_SD[idx] <- val_sd_se
    std_Estimate_CI[idx] <- sprintf("%.3f (%.3f, %.3f)%s", val_mean_se, val_mean_se - 1.96*val_sd_se, val_mean_se + 1.96*val_sd_se, get_bf_stars(val_mean_se, val_sd_se))
    idx <- idx + 1
    
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
  
  SE_list <- list() 
  latent_df_list <- list()
  
  if(K > 1) {
    for(k in 1:K) {
      opt_hess <- optim(par = as.vector(B_list[[k]]), fn = obj_k, X = X, Y = Y_imputed,
                        w_k = gamma_ik[, k], Sigma = Sigma_list[[k]], Lambda_eps = Lambda_eps,
                        q = q, p = p, method = "BFGS", hessian = TRUE)
      
      Cov_B_k <- tryCatch({ solve(opt_hess$hessian) }, error = function(e) { solve(opt_hess$hessian + diag(1e-5, p*q)) })
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
        val_mean_se <- sqrt(Sigma_list[[k]][j, j])
        val_sd_se <- val_mean_se / sqrt(2 * N_k)
        lat_Output[idx] <- col_Y[j]
        lat_Variable[idx] <- "Standard Error"
        lat_Mean[idx] <- val_mean_se
        lat_SD[idx] <- val_sd_se
        lat_Estimate_CI[idx] <- sprintf("%.3f (%.3f, %.3f)%s", val_mean_se, val_mean_se - 1.96*val_sd_se, val_mean_se + 1.96*val_sd_se, get_bf_stars(val_mean_se, val_sd_se))
        idx <- idx + 1
        
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
  # Part E: Error Table (Variance Contraction Profiler)
  # ---------------------------------------------------------
  make_error_spacer_row <- function() {
    data.frame(Class="", Output="", Variable="", Prior_Mean=NA, Post_Mean=NA,
               Mean_Diff=NA, Coeff_Var=NA, Prior_SD=NA, Post_SD=NA, SD_Ratio=NA,
               Post_95_CI="", check.names=FALSE, stringsAsFactors=FALSE)
  }
  
  error_df_list <- list()
  if(K > 1) {
    for(k in 1:K) {
      SE_B_k <- SE_list[[k]]
      n_err_rows <- q * (p + 1)
      class_prop_label <- sprintf("%d (%.3f)", k, pi_k[k])
      
      err_Class <- rep(class_prop_label, n_err_rows)
      err_Output <- rep("", n_err_rows)
      err_Variable <- rep("", n_err_rows)
      err_Prior_Mean <- rep(0, n_err_rows)
      err_Post_Mean <- rep(0, n_err_rows)
      err_Mean_Diff <- rep(0, n_err_rows)
      err_Coeff_Var <- rep(NA, n_err_rows)
      err_Prior_SD <- rep(0, n_err_rows)
      err_Post_SD <- rep(0, n_err_rows)
      err_SD_Ratio <- rep(0, n_err_rows)
      err_Post_95_CI <- rep("", n_err_rows)
      
      idx <- 1
      for(j in 1:q) {
        # 1. ADD ROW FOR RESIDUAL SD
        prior_sd_resid <- sqrt(Sigma_null[j, j])
        post_sd_resid  <- sqrt(Sigma_list[[k]][j, j])
        
        err_Output[idx] <- col_Y[j]
        err_Variable[idx] <- "Residual SD"
        err_Prior_Mean[idx] <- prior_sd_resid
        err_Post_Mean[idx] <- post_sd_resid
        err_Mean_Diff[idx] <- post_sd_resid - prior_sd_resid
        err_Prior_SD[idx] <- prior_sd_resid
        err_Post_SD[idx] <- post_sd_resid
        err_SD_Ratio[idx] <- post_sd_resid / prior_sd_resid
        err_Post_95_CI[idx] <- "" 
        idx <- idx + 1
        
        # 2. ADD ROWS FOR COVARIATE COEFFICIENTS
        prior_sd_beta <- sqrt(Sigma_null[j, j]) / sqrt(prior_lambda)
        
        for(i in 1:p) {
          prior_m  <- 0.0  
          post_m   <- B_list[[k]][i, j]
          post_sd  <- SE_B_k[i, j]
          
          m_diff <- post_m - prior_m
          cv_val <- if (post_m == 0) Inf else post_sd / post_m
          sd_ratio <- post_sd / prior_sd_beta
          
          err_Output[idx] <- col_Y[j]
          err_Variable[idx] <- col_X[i]
          err_Prior_Mean[idx] <- prior_m
          err_Post_Mean[idx] <- post_m
          err_Mean_Diff[idx] <- m_diff
          err_Coeff_Var[idx] <- cv_val
          err_Prior_SD[idx] <- prior_sd_beta
          err_Post_SD[idx] <- post_sd
          err_SD_Ratio[idx] <- sd_ratio
          err_Post_95_CI[idx] <- sprintf("(%.3f, %.3f)", post_m - 1.96 * post_sd, post_m + 1.96 * post_sd)
          
          idx <- idx + 1
        }
      }
      error_df_list[[k]] <- data.frame(
        Class = err_Class, Output = err_Output, Variable = err_Variable,
        Prior_Mean = err_Prior_Mean, Post_Mean = err_Post_Mean, Mean_Diff = err_Mean_Diff,
        Coeff_Var = err_Coeff_Var, Prior_SD = err_Prior_SD, Post_SD = err_Post_SD,
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
  # Part F: Visualizations (Conditional Trajectory Mapping)
  # ---------------------------------------------------------
  Y_mat <- as.matrix(Y_imputed)
  X_mat <- as.matrix(X)
  cluster_assignment <- factor(apply(gamma_ik, 1, which.max), levels = 1:K)
  
  non_intercept_idx <- which(apply(X_mat, 2, function(x) length(unique(x)) > 1))
  col_X_plot <- colnames(X_mat); if(is.null(col_X_plot)) col_X_plot <- paste0("X", 1:p)
  col_Y_plot <- colnames(Y_mat); if(is.null(col_Y_plot)) col_Y_plot <- paste0("Y", 1:q)
  
  obj_global <- function(B_vec, X, Y) {
    B_mat <- matrix(B_vec, nrow = p, ncol = q)
    g_val <- g_func(X, B_mat)
    if(any(!is.finite(g_val))) return(1e20)
    sum((Y - g_val)^2)
  }
  opt_global <- optim(par = as.vector(B_std_init), fn = obj_global, X = X_mat, Y = Y_mat, method = "BFGS")
  beta_ols <- matrix(opt_global$par, nrow = p, ncol = q)
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
      y_pred_ols <- g_func(X_pred_ols, beta_ols)[, d]
      df_ols <- data.frame(X = x_grid, Y = y_pred_ols)
      
      pred_list <- list()
      for(k in 1:K) {
        sigma_kd <- sqrt(Sigma_list[[k]][d, d])
        X_pred_k <- matrix(rep(X_mean, each = length(x_grid)), nrow = length(x_grid))
        X_pred_k[, j] <- x_grid
        y_pred_k <- g_func(X_pred_k, B_list[[k]])[, d]
        
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
