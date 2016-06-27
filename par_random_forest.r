
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
myFiles = myFiles2 #[grep("g1|g2|g3|g4",myFiles2)]

a=read.csv(paste(getwd(),'/',myFiles[1],sep=''),as.is=T)

##creating a matrix of all the possible combinations of parameters we want to run
rf_parameters=expand.grid(ntree = seq(100,1800,100), mtry = c(floor((ncol(a)-3)/6),floor((ncol(a)-3)/3),floor((ncol(a)-3)*1.5)), nodesize=5)



library(randomForest)
library(doSNOW)  
library(foreach)  

##For user's ease, a progress bar would show how many models have been created
pb = winProgressBar(title = paste("RF Progress Bar"), min = 0, max = length(myFiles), width = 400)

for(j in 1:length(myFiles))
	{
		a=read.csv(paste(preproc_dir,'/',myFiles[j],sep=''),as.is=T)
		colnames(a)=paste(gsub(".","",colnames(a),fixed=TRUE))
		a$dv=gsub('0','r',a$dv)
		a$dv=gsub('1','s',a$dv)
		val_data= read.csv(paste(preproc_dir,'/',"val.csv",sep=''),as.is=T)
		colnames(val_data)=paste(gsub(".","",colnames(val_data),fixed=TRUE))
		val_data=val_data[,colnames(a)]
		test_data= read.csv(paste(preproc_dir,'/',"test.csv",sep=''),as.is=T)
		colnames(test_data)=paste(gsub(".","",colnames(test_data),fixed=TRUE))
		test_data=test_data[,colnames(a)]

			

		cl=makeCluster(3) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		rf_model = foreach(k = 1:nrow(rf_parameters),.packages="randomForest") %dopar% {
							randomForest(x = a[,-c(1:3)],y =  as.factor(a[,2]), ntree=rf_parameters[k,1],mtry=rf_parameters[k,2],nodesize=rf_parameters[k,3])
						}
		stopCluster(cl)

		cl=makeCluster(2) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		rf_val = foreach(k = 1:nrow(rf_parameters), .combine='cbind', .packages="randomForest") %dopar% 
				{
				predict(rf_model[[k]], val_data[,-c(1:3)], type="prob")[,2]   ##using models to predict validation sample values
				}                
		stopCluster(cl)
		
		val_out_local=as.data.frame(rf_val)
		colnames(val_out_local) = paste(gsub(".","_",colnames(rf_val),fixed=TRUE))
		colnames(val_out_local) = paste(gsub("result",paste("rf_val",j,sep="_"),colnames(rf_val),fixed=TRUE))
		if(j==1)
			{rf_val_full=val_out_local
		}else
		{rf_val_full=cbind(rf_val_full,val_out_local)}
	
		cl=makeCluster(2) #change the 2 to your number of CPU cores  
		registerDoSNOW(cl)  

		rf_test = foreach(k = 1:nrow(rf_parameters), .combine='cbind', .packages="randomForest") %dopar% 
				{
				predict(rf_model[[k]], test_data[,-c(1:3)], type="prob")[,2] ##using models to predict test sample values
				}                
		stopCluster(cl)
	
		test_out_local=as.data.frame(rf_test)
		colnames(test_out_local) = paste(gsub(".","_",colnames(rf_test),fixed=TRUE))
		colnames(test_out_local) = paste(gsub("result",paste("rf_test",j,sep="_"),colnames(rf_test),fixed=TRUE))
		if(j==1)
			{rf_test_full=test_out_local
		}else
		{rf_test_full=cbind(rf_test_full,test_out_local)}
	setWinProgressBar(pb,j, title=paste(" RF :", round(j/length(myFiles)*100, 0),"% done"))
		
	}
	
	setwd(candidates_val)
	write.csv(rf_val_full,"rf_val.csv",row.names=F)
	
	setwd(candidates_test)
	write.csv(rf_test_full,"rf_test.csv",row.names=F)
	
close(pb)
