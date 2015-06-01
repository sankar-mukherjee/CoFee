datasets = contains two datasets CoFee-BigDataset_ordered and allmerged_ordered. their 5 imputed datasets are big and out respectively.

generate_imputation.R = generate multiple imputation with EM algorithm to overcome NaN and Inf values
findanyrealtion.R = plotting between different pos features and acoustic features
rChartCode.R = interactive R plotting
Rplot.png = missmap of the allmerged_ordered dataset by Amelia program

javascript.js = generated rChart (HighCharts) from findanyrealtion.R with this command


h1 <- hPlot(x = "do", y = "duration", data = data, type = c("line", 
    "bubble", "scatter"), group = "Clap", size = "Age")
h1$print("chart5")