% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/SmallFuncs.R
\name{RandParams}
\alias{RandParams}
\title{Generate Random Parameters}
\usage{
RandParams(x, Rpoints, bounds)
}
\arguments{
\item{x}{Parameter Name}

\item{Rpoints}{number of parameter sets to create.}

\item{bounds}{original bounds list}
}
\value{
a data.table of random parameters
}
\description{
Generates a list of random parameters within bounds
}
\keyword{internal}
