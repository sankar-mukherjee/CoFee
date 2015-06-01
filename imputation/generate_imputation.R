# generate multiple imputation for NaN and inf values using amelia packege
require(Amelia)

data = read.csv('datasets/CoFee-BigDataset_ordered.csv',sep=';')

# remove columns with textual value
new_data = subset(data, select=-c(prevfirstBi, rol, simple, role, othLastBi, oth, prevLastBi, corpus, IPUInterl, trans, othFirstTok, tokenInterl, token, othfirstBi, speaker, othLastTok, prevLastTok, interlocutor, uttInterl, prevFirstTok, gest))

#replace Nan and Inf with NA
new_data <- replace(new_data, is.na(new_data), NA)
new_data <- do.call(data.frame,lapply(new_data, function(x) replace(x, is.infinite(x),NA)))

#remove unnecessary coulms as per amelia program (because of the sametype of value)
new_data <- new_data[,!names(new_data) %in% c("aperiodAVDiff")]
new_data <- new_data[,!names(new_data) %in% c("TFcomparedIPU")]
new_data <- new_data[,!names(new_data) %in% c("TFcomparedWord")]

# empri = Ridge Priors for Large Correlations (5 times)
a.out <- amelia(new_data, m = 5,  cs = "spk", idvars = c("id"), p2s = 2, empri = .05*nrow(new_data))

# visualize
missmap(a.out)
plot(a.out, which.vars = 3)
compare.density(a.out, var = "intQ3Interl",legend=FALSE)
overimpute(a.out,var = "intQ3Interl")
disperse(a.out, dims = 2, m = 5)

#this will not work
tscsPlot(a.out, cs = "spk", main = "spk", var = "form3", ylim = c(-10, 60))

#write to CSV
write.amelia(obj=a.out, file.stem = "datasets/big")
