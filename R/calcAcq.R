#' @title Calculate Acquisition Function
#'
#' @description
#' Function to be Maximized
#'
#' @param Params Parameter set to predict
#' @param GP an object of class gp or gp.list
#' @param acq Acquisition function type to be used
#' @param y_max The current maximum known value of the target utility function
#' @param kappa tunable parameter kappa to balance exploitation against exploration
#' @param eps tunable parameter epsilon to balance exploitation against exploration
#' @importFrom stats dnorm pnorm
#' @return The acquisition function value.
#' @keywords internal
#' @export

calcAcq <- function(Params, GP, acq, y_max, kappa, eps) {
  # Utility Function Type
  # https://www.cse.wustl.edu/~garnett/cse515t/spring_2015/files/lecture_notes/12.pdf

  if (class(GP)[[1]] == "list") {
    GPs <- GP[[1]]
    GPe <- GP[[2]]
  } else{
    GPs <- GP
  }

  if (acq == "ucb") {

    GP_Pred <- GPs$predict((Params), se.fit = TRUE)
    return((GP_Pred$mean + kappa * (GP_Pred$se)))

  } else if (acq == "ei") {

    GP_Pred <- GPs$predict((Params), se.fit = TRUE)
    z <- (GP_Pred$mean - y_max - eps) / (GP_Pred$se)
    return(((GP_Pred$mean - y_max - eps) * pnorm(z) + (GP_Pred$se) * dnorm(z)))

  } else if (acq == "eips") {

    GPs_Pred <- GPs$predict((Params), se.fit = TRUE)
    GPe_Pred <- GPe$predict((Params), se.fit = TRUE)
    z <- (GPs_Pred$mean - y_max - eps) / (GPs_Pred$se)
    return(((GPs_Pred$mean - y_max - eps) * pnorm(z) + (GPs_Pred$se) * dnorm(z))/GPe_Pred$mean)

  } else if (acq == "poi") {

    GP_Pred <- GPs$predict((Params), se.fit = TRUE)
    z <- (GP_Pred$mean - y_max - eps) / (GP_Pred$se)
    return((pnorm(z)))

  }

}


