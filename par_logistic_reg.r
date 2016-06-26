
rm(list=ls())

user=(Sys.info()[6])
Desktop=paste("C:/Users/",user,"/Desktop/",sep="")
setwd(Desktop)


home=paste(Desktop,"MEMS/S4/R Programming/German/",sep="")

setwd(paste(home,'Model',sep=''))

preproc_dir=paste(home,'Model',"/","preproc_data",sep='')
dir.create("candidates_val")
candidates_val=paste(home,'Model',"/","candidates_val",sep='')
dir.create("candidates_test")
candidates_test=paste(home,'Model',"/","candidates_test",sep='')
	
### install.packages('ggplot2',repos='http://mirrors.softliste.de/cran/',dependencies=T)
library(caret)
library(gbm)
library(nnet)
library(pROC)
library(hmeasure) 
library(dplyr)
library(randomForest)
library(doSNOW)  
library(foreach)  


	setwd(preproc_dir)
	myFiles2 <- list.files(pattern="dfold.*csv")
	myFiles <- myFiles2 #[grep("g1|g2|g3|g4",myFiles2)]

		
	##### Logistic Regrssion
	library(stepPlr)
	setwd(preproc_dir)
	
	lr_parameters=expand.grid(lambda = 2^seq(-19,6,2), cp = c("aic","bic"))
		# lambda	
		# regularization parameter for the L2 norm of the coefficients. The minimizing criterion in plr is -log-likelihood+λ*\|β\|^2. Default is lambda=1e-4.
		# cp	
		# complexity parameter to be used when computing the score. score=deviance+cp*df. If cp="aic" or cp="bic", these are converted to cp=2 or cp=log(sample size).
		# lr_parameters=lr_parameters[1:5,]
pb <- winProgressBar(title = paste("LogR Progress Bar"), min = 0, max = length(myFiles), width = 400)

# for (j in 1:5)
	for (j in 1:length(myFiles))
	{
		
		a=read.csv(paste(preproc_dir,'/',myFiles[j],sep=''),as.is=T)
		colnames(a)<-paste(gsub(".","",colnames(a),fixed=TRUE))
		val_data= read.csv(paste(preproc_dir,'/',"val.csv",sep=''),as.is=T)
		colnames(val_data)<-paste(gsub(".","",colnames(val_data),fixed=TRUE))
		val_data=val_data[,colnames(a)]
		test_data= read.csv(paste(preproc_dir,'/',"test.csv",sep=''),as.is=T)
		colnames(test_data)<-paste(gsub(".","",colnames(test_data),fixed=TRUE))
		test_data=test_data[,colnames(a)]
		

		cl<-makeCluster(3) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		lr_model <- foreach(k = 1:nrow(lr_parameters),.packages="stepPlr") %dopar% {

			
			lr_model <- plr(x = a[,-c(1:3)],y = as.numeric(a[,2]),lambda=lr_parameters[k,1], cp=lr_parameters[k,2])
						}
		stopCluster(cl)

		cl<-makeCluster(3) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		lr_val <- foreach(k = 1:nrow(lr_parameters), .combine='cbind', .packages="stepPlr") %dopar% {
				predict(lr_model[[k]],val_data[,-c(1:3)],type="response")
						
					}                
		stopCluster(cl)
		
		val_out_local=as.data.frame(lr_val)
		colnames(val_out_local) <- paste(gsub(".","_",colnames(lr_val),fixed=TRUE))
		colnames(val_out_local) <- paste(gsub("result",paste("lr_val",j,sep="_"),colnames(lr_val),fixed=TRUE))
		if(j==1)
			{lr_val_full=val_out_local
		}else
		{lr_val_full=cbind(lr_val_full,val_out_local)}
	
		cl<-makeCluster(3) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		lr_test <- foreach(k = 1:nrow(lr_parameters), .combine='cbind', .packages="stepPlr") %dopar% {
				predict(lr_model[[k]],test_data[,-c(1:3)],type="response")
						}                
		stopCluster(cl)
	
		test_out_local=as.data.frame(lr_test)
		colnames(test_out_local) <- paste(gsub(".","_",colnames(lr_test),fixed=TRUE))
		colnames(test_out_local) <- paste(gsub("result",paste("lr_test",j,sep="_"),colnames(lr_test),fixed=TRUE))
		if(j==1)
			{lr_test_full=test_out_local
		}else
		{lr_test_full=cbind(lr_test_full,test_out_local)}
	setWinProgressBar(pb,j, title=paste(" LogR :", round(j/length(myFiles)*100, 0),"% done"))
		
	}
	
	setwd(candidates_val)
	write.csv(lr_val_full,"lr_val.csv",row.names=F)
	
	setwd(candidates_test)
	write.csv(lr_test_full,"lr_test.csv",row.names=F)
	
	setwd(home)
close(pb)


