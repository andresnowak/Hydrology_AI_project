# Dataset Info

## General Info

Dataset is not big data, because we have a small amount of data and we are not continuously receiving data.

### Size

There are 42059 observations and 59 features

### Null values

There are no Null values in the PlatteRiver csv Dataset

### Working features

From our 59 features

-   2 are our answer features (stage and discharge)
-   stage and discharge are highly correlated but we are still going to leave them.
-   5 features are not features for this dataset (Filename, Agency, SiteNumber, TimeZone, CalcTimestamp)
-   2 features (Width and Height) are features that are highly correlated and don't add info to our dataset because (there are only two distinct values)
-   areaFeatCount and WwCurveLineMin are features that have no linear relationship (pearson correlation) so they can be discarded
-   17 features are highly correlated (height, isoSpeed, entropySigma, hMean, hSigma, entropySigma0, hMean0, hSigma0, entropySigma1, hMean1, hSigma1, WeirPt1Y, WeirPt2X, WeirPt2Y, WwCurveLineMax, WwCurveLineMean, WwCurveLineSigma)
-   The values fro Stage and Discharge seem to have seasonality, but Dickey-Fuller test tells us there is no seasonality and Kpss tells us there is a seasonality, this could be because it is true, because the data is not equidistant in time or maybe because of how unpredictable climate is.
