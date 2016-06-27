
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
	
##loading the names of file in the directory
setwd(preproc_dir)
myFiles2 = list.files(pattern="dfold.*csv")
myFiles = myFiles2 

		
##### Logistic Regrssion in parallel processing
library(stepPlr)
library(doSNOW)  
library(foreach)  

setwd(preproc_dir)


##creating a matrix of all the possible combinations of parameters we want to run
lr_parameters=expand.grid(lambda = 2^seq(-19,6,2), cp = c("aic","bic"))

##For user's ease, a progress bar would show how many models have been created
pb = winProgressBar(title = paste("LogR Progress Bar"), min = 0, max = length(myFiles), width = 400)


for (j in 1:length(myFiles))
	{
		
		a=read.csv(paste(preproc_dir,'/',myFiles[j],sep=''),as.is=T)
		colnames(a)=paste(gsub(".","",colnames(a),fixed=TRUE))   ##fixing column names
		val_data= read.csv(paste(preproc_dir,'/',"val.csv",sep=''),as.is=T)
		colnames(val_data)=paste(gsub(".","",colnames(val_data),fixed=TRUE))
		val_data=val_data[,colnames(a)]
		test_data= read.csv(paste(preproc_dir,'/',"test.csv",sep=''),as.is=T)
		colnames(test_data)=paste(gsub(".","",colnames(test_data),fixed=TRUE))
		test_data=test_data[,colnames(a)]
		

		cl=makeCluster(3) #change the 3 to your number of CPU cores  
		registerDoSNOW(cl)  

		lr_model = foreach(k = 1:nrow(lr_parameters),.packages="stepPlr") %dopar% 
				{
				lr_model = plr(x = a[,-c(1:3)],y = as.numeric(a[,2]),lambda=lr_parameters[k,1], cp=lr_parameters[k,2]) ##building models
				}
		stopCluster(cl)

		cl=makeCluster(3) #change the 3 to your number of CPU cores  
		registerDoSNOW(cl)  

		lr_val = foreach(k = 1:nrow(lr_parameters), .combine='cbind', .packages="stepPlr") %dopar% 
				{
				predict(lr_model[[k]],val_data[,-c(1:3)],type="response")  ##using models to predict validation sample values		
				}                
		stopCluster(cl)
		
		val_out_local=as.data.frame(lr_val)
		colnames(val_out_local) = paste(gsub(".","_",colnames(lr_val),fixed=TRUE)) ##fixing column names
		colnames(val_out_local) = paste(gsub("result",paste("lr_val",j,sep="_"),colnames(lr_val),fixed=TRUE))
		
		if(j==1)
			{lr_val_full=val_out_local
		}else
		{lr_val_full=cbind(lr_val_full,val_out_local)}
	
		cl=makeCluster(3) #change the 3 to your number of CPU cores  
		registerDoSNOW(cl)  

		lr_test = foreach(k = 1:nrow(lr_parameters), .combine='cbind', .packages="stepPlr") %dopar% 
				{
				predict(lr_model[[k]],test_data[,-c(1:3)],type="response")  ##using models to predict test sample values
				}                
		stopCluster(cl)
	
		test_out_local=as.data.frame(lr_test)
		colnames(test_out_local) = paste(gsub(".","_",colnames(lr_test),fixed=TRUE))   ##fixing column names
		colnames(test_out_local) = paste(gsub("result",paste("lr_test",j,sep="_"),colnames(lr_test),fixed=TRUE))
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


