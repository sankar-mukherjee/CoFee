# plotting different features to find any correlation. Using imputed data to get full acoustic advantage. Labels are collected from original datasets.
require(ggplot2)

data = read.csv('datasets/big1.csv',head=TRUE,sep=',')
ori = read.csv('datasets/CoFee-BigDataset_ordered.csv',head=TRUE,sep=';')

plot(ori$simple)
#################### sa #########################

data$sa[data$sa > 4]=4
data$sa[data$sa < 0]=0

plot(density(data$sa))

plot(density(data$form2),xlim = c(0,600))
plot(hist(data$form3))
plot(data$sa,data$form3)

plot(density(data$span))

plot(data$sa,data$span)
qplot(data$sa,data$form2,col=ori$simple)
qplot(data$sa,data$SimScoreIPU,col=ori$simple)
qplot(data$sa,data$SimScoreWord,col=ori$simple)
qplot(data$sa,data$pauseBefore,col=ori$simple,xlim = c(0,10),ylim=c(0,10))
qplot(data$sa,data$height,col=ori$simple,xlim = c(0,10),ylim=c(0,10))

#################### osa #########################
data$osa[data$osa > 4]=4
data$osa[data$osa < 0]=0
plot(density(data$osa))

plot(data$osa,data$duration,pch = '*')

plot(density(data$do))
plot(data$do,data$height,pch = '*',xlim = c(0,4))
qplot(data$do,data$duration,xlim = c(0,4),col=ori$simple)

index= which(ori$simple == 'complex')
x = data$do[-index]
y = data$duration[-index]
label = ori$simple[-index]
qplot(x,y,xlim = c(0,1),ylim=c(0,3),color=label)

plot1 <- mPlot(x="do", y="duration", type="Line", data=data)

h1 <- hPlot(x="do", y="duration", data = data)
sink("javascript.htm")
h1$print("chart5")
sink()

#################### pa #########################
data$pa[data$pa > 4]=4
data$pa[data$pa < 0]=0
plot(density(data$pa))
qplot(data$pa,data$form3,data=data,color=ori$simple)
qplot(data$pa,data$aperiodAV,data=data,color=ori$simple)

################## pb ##########################
data$pb[data$pb > 4]=4
data$pb[data$pb < 0]=0
plot(density(data$pb))
qplot(data$pb,data$form3,color=ori$simple)
qplot(data$pb,data$intQ1raw,color=ori$simple)

################## opa ##########################
data$opa[data$opa > 4]=4
data$opa[data$opa < 0]=0
plot(density(data$opa))
qplot(data$opa,data$intQ2Interl,color=ori$simple)


#################### osb #########################
data$osb[data$osb > 4]=4
data$osb[data$osb < 0]=0
plot(density(data$osb))
qplot(data$osa,data$intQ1,color=ori$simple)





