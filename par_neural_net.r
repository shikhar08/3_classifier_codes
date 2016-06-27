
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
		
	


setwd(preproc_dir)
myFiles2 = list.files(pattern="dfold.*csv")
myFiles = myFiles2 # [grep("g1|g2|g3|g4",myFiles2)]

		
#### ANN Model
	

library(nnet)
library(doSNOW)  
library(foreach)  


##creating a matrix of all the possible combinations of parameters we want to run
ann_parameters=expand.grid(size =seq(1,15,2), decay=10^seq(-4,0,2),maxit=c(100,200,1000))

##For user's ease, a progress bar would show how many models have been created

pb = winProgressBar(title = paste("ANN Progress Bar"), min = 0, max = length(myFiles), width = 400)



for (j in 1:length(myFiles))
	{

		a=read.csv(paste(preproc_dir,'/',myFiles[j],sep=''),as.is=T)
		colnames(a)=paste(gsub(".","",colnames(a),fixed=TRUE)) ##fixing column names
		val_data= read.csv(paste(preproc_dir,'/',"val.csv",sep=''),as.is=T)
		colnames(val_data)=paste(gsub(".","",colnames(val_data),fixed=TRUE))
		val_data=val_data[,colnames(a)]
		test_data= read.csv(paste(preproc_dir,'/',"test.csv",sep=''),as.is=T)
		colnames(test_data)=paste(gsub(".","",colnames(test_data),fixed=TRUE))
		test_data=test_data[,colnames(a)]
			

		cl=makeCluster(3) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		nn_model = foreach(k = 1:nrow(ann_parameters),.packages="nnet") %dopar% 
					{
					nnet(x = a[,-c(1:3)], y = (a[,2]), size=ann_parameters[k,1], decay=ann_parameters[k,2], maxit=ann_parameters[k,3]) ##building models
					}
		stopCluster(cl)

		cl=makeCluster(2) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		ann_val = foreach(k = 1:nrow(ann_parameters), .combine='cbind', .packages="nnet") %dopar% 
					{
					predict(nn_model[[k]],val_data[,-c(1:3)])  ##using models to predict validation sample values
					}                
		stopCluster(cl)
		
		val_out_local=as.data.frame(ann_val)
		z1=paste(expand.grid(l=paste("ann_val_",j,sep=""),m=c(1:nrow(ann_parameters)))[,1],expand.grid(l=paste("ann_val_",j,sep=""),m=c(1:nrow(ann_parameters)))[,2],sep="_")
		colnames(val_out_local) = paste(z1) ##fixing column names
		
		if(j==1)
			{ann_val_full=val_out_local
		}else
		{ann_val_full=cbind(ann_val_full,val_out_local)}
	
		cl=makeCluster(2) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		ann_test = foreach(k = 1:nrow(ann_parameters), .combine='cbind', .packages="nnet") %dopar% 
					{
					predict(nn_model[[k]],test_data[,-c(1:3)]) ##using models to predict test sample values
					}                
		stopCluster(cl)
	
		test_out_local=as.data.frame(ann_test)
		z2=paste(expand.grid(l=paste("ann_test_",j,sep=""),m=c(1:nrow(ann_parameters)))[,1],expand.grid(l=paste("ann_test_",j,sep=""),m=c(1:nrow(ann_parameters)))[,2],sep="_")
		colnames(test_out_local) = paste(z2) ##fixing column names
		if(j==1)
			{ann_test_full=test_out_local
		}else
		{ann_test_full=cbind(ann_test_full,test_out_local)}
	setWinProgressBar(pb,j, title=paste(" ANN :", round(j/length(myFiles)*100, 0),"% done"))
		
	}
	
	setwd(candidates_val)
	write.csv(ann_val_full,"ann_val.csv",row.names=F)
	
	setwd(candidates_test)
	write.csv(ann_test_full,"ann_test.csv",row.names=F)
	
	setwd(home)
	close(pb)
 
 
 