PMT <- function(xtrain,ytrain,w,depth=5,min_size=20,M=10){
  
  base_tree <- rpart(factor(ytrain)~.,data=xtrain,weights=w,control=list(maxdepth=depth,minsplit =min_size))
  
  terminal_nodes <- sort(unique(base_tree$where)) #pates0('n',sort(unique(base_tree$where))
  
  pbts <- list()#vector("list",length = length(terminal_nodes))
  
  for(node in terminal_nodes){
    index_node <- as.numeric(which(base_tree$where == node))
    #print(index_node)
    
    x_node <- xtrain[index_node,]
    y_node <- ytrain[index_node]
    names(y_node)<- row.names(x_node)
    w_node <- w[index_node]
    if(length(unique(y_node))==1){
      
      pbts[[paste0("n",node)]] <- list(predicted=unique(y_node),unique=TRUE,fitted=y_node)
    }else{
      
      fit <- WoProbitBoost(x_node, y_node, Wo=w_node, M_max = M)
      names(fit$fitted_p)<- row.names(x_node)
      pbts[[paste0("n",node)]] <-list(boost_feats=fit$boost_feat,local_nodes=index_node,fitted=fit$fitted_p, ynode=y_node,unique=FALSE)#MWProbitBoost(x_node, y_node, Wo=w_node,M_max=M,aic=aic)
    }
    
    
    #LogitBoost(x_node, y_node, nIter = ncol(x_node)) #WoProbitBoost(x_node, y_node, Wo=w_node,M_max=M,aic=aic)$boost_feat #PB(x_node,y_node,M=M,aic=aic,depth = depth,min_size = min_size)
    
  }
  #object <- list(pbt=pbts,terminal_nodes=terminal_nodes)
  
  return(list(tree=base_tree,pbt=pbts,terminal_nodes=terminal_nodes))
}




pmt.predict<- function(pmt_list,new_data){
  
  tree <- pmt_list$tree
  ter_nodes <- pmt_list$terminal_nodes
  
  #pbts <- pmt_list$pbt
  
  pred_nodes_index <- rpart:::pred.rpart(tree, rpart:::rpart.matrix(new_data))
  pred_nodes <- paste0("n",as.numeric(pred_nodes_index))
  
  predicted <- predictProbit(as.matrix(new_data),pmt_list,pred_nodes) #numeric(length = length(pred_nodes))
  
  
  return(predicted)
}


AdaPMT <- function(ms=10,M=5,depth=5,trainx,trainy,xtest,step=0.5,w_init,tpx_init,size=15,lab_list){
  n <- length(trainy)
  
  tpx <- tpx_init
  cpx <- tpx
  w <- w_init
  px <- numeric(n)
  re <- 0
  n_Class <- length(unique(trainy))
  Cx <- matrix(0,ncol=n_Class,nrow(xtest))
  lab_list <- lab_list#as.vector(sort(unique(trainy)))
  if(n_Class==2){
    for(m in 1:ms){
      # print(w)

      pmt <- PMT(trainx,trainy,w=w,depth=depth,M=M,min_size=size)
      terminode <- paste0("n",pmt$terminal_nodes)
      
      pred <- ifelse(do.call("c",lapply(terminode,function(node) pmt$pbt[[node]]$fitted ))>0.5,1,0)

      #print(Ix)
      #print(pred[names(trainy)])
      # pred <- ifelse(pmt.predict2(pmt,trainx)>0.5,1,0)
      Ix <- ifelse(pred[names(trainy)]!=trainy,1,0)
      err <- sum(w*Ix)/(sum(w))
      
      
      alpha <-step*log((1-err+1e-24)/(err+1e-24))
      
      w <- w*exp(alpha*Ix)
      w <- w/sum(w)
      # print(pmt$pbt)
      
      
      test <- pmt.predict(pmt,xtest)
      # print(test)
      
      test <- ifelse(test>0.5,1,-1)  
      
      tpx <-  tpx +test*alpha
      
      #tpx <-  tpx +test*alpha
      #  print(m)
      
    } 
    return(tpx)
  }else{
    
    for(m in 1:ms){
      boost_feat_train <- matrix(-Inf,ncol=n_Class,nrow=n)   
      boost_feat_test<- matrix(-Inf,ncol=n_Class,nrow=nrow(xtest))   # ... recursivly
      for (jClass in 1:n_Class) {
        #print('loop')
        y = as.numeric(trainy==lab_list[jClass]) # lablist[jClass]->1; rest->0
        pmt <- PMT(trainx,y,w=w,depth=depth,M=M,min_size=size)
        #pred <- pmt.predict(pmt,trainx,K2)
        
        terminode <- paste0("n",pmt$terminal_nodes)
        pj <- do.call("c",lapply(terminode,function(node) pmt$pbt[[node]]$fitted ))
        boost_feat_train[,jClass] <- pj[names(trainy)]# ifelse(do.call("c",lapply(terminode,function(node) pmt$pbt[[node]]$fitted ))>0.5,1,0)#pmt.predict2(pmt,trainx)#pmt$predict(trainx)
        
        boost_feat_test[,jClass] <- pmt.predict(pmt,xtest)
      }
      pred <- apply(boost_feat_train,1,function(row) lab_list[which.max(row)] )
      # print(boost_feat_train)
      #print(pred)
      #  pred <- ifelse(pmt$predict(trainx)>0,lablist[2],lablist[1])
      Ix <- ifelse(pred!=trainy,1,0)
      
      err <- sum(w*Ix)/(sum(w))
      
      
      alpha <-step*log((1-err+1e-24)/(err+1e-24))+log(n_Class-1)
      
      w <- w*exp(alpha*Ix)
      w <- w/sum(w)
      
      test <- apply(boost_feat_test,1,function(row) which.max(row) )#ifelse(pmt$predict(xtest)==1,1,-1)  
      
      
      for(row in 1:nrow(xtest)){
        
        Cx[row,test[row]] <- Cx[row,test[row]] +alpha
      }
      
      #print(m)
      
      
    }
    
    
    return(Cx)
    
    
    
  }
  
}



SBPMT <- function(n_tree=20,n_iteration=5,M=5,depth=5,xtrain,ytrain,xtest,step=0.5,size=15,alpha=0.7,seed=NULL){
  
  # n_tree <- n_tree
  # n_iteration <- n_iteration
  # 
  if(is.null(xtest)){
    xtest=xtrain
  }
  lab_list = as.vector(sort(unique(ytrain)))
  if(length(unique(ytrain))==2){
    
    rf_test <- matrix(0,ncol=n_tree,nrow=nrow(xtest))#$numeric(nrow(xtest))
    hist_accs <- numeric(n_tree)
    if(!is.null(seed)){
      set.seed(42)
    }

    for(n in 1:n_tree){
      
      samp_index <-  sample(1:nrow(xtrain),size=ceiling(alpha*nrow(xtrain)),replace = FALSE) #sample(1:nrow(xtrain),replace = TRUE)
      samp_fea <- 1:ncol(xtrain)#sample(1:ncol(xtrain),size=floor(sqrt(ncol(xtrain))),replace = FALSE)
      usp <- unique(samp_index)
      #print(length(usp))
      x_rf <- xtrain[samp_index,]
      y_rf <- ytrain[samp_index]
      
      w_init <- rep(1/length(usp),length(usp))
      
      tpx_init <- numeric(nrow(xtest))
      
      
      tpx_rf <- AdaPMT(ms=n_iteration,M=M,depth=depth,trainx=x_rf,trainy=y_rf,xtest=xtest,step=step,tpx_init=tpx_init,w_init=w_init,size=size,lab_list=lab_list)
      rf_test[,n] <- ifelse(tpx_rf>0,1,-1)
      
      
    }
    pred_binary <- apply(rf_test,1,function(row) ifelse(sum(row)>0,lab_list[2],lab_list[1]))
    return(pred_binary)
    
  }else if(length(unique(ytrain))>2){#multi-class, one-vs-all strategy used
    
    
    rf_test <- array(0,dim=c(n_tree,nrow(xtest),length(unique(ytrain))))#matrix(0,ncol=n_tree,nrow=nrow(xtest))#$numeric(nrow(xtest))
    hist_accs <- numeric(n_tree)
    
    
    for(n in 1:n_tree){
      
      samp_index <- sample(1:nrow(xtrain),size=alpha*nrow(xtrain),replace = FALSE)#sample(1:nrow(xtrain),replace = TRUE)
      samp_fea <-   sample(1:ncol(xtrain),size=floor(sqrt(ncol(xtrain))),replace = FALSE)#1:ncol(xtrain) #unique(sample(1:ncol(xtrain),replace = TRUE))#unique(sample(1:ncol(xtrain),replace = TRUE)
      usp <- unique(samp_index)
      #print(length(usp))
      x_rf <- xtrain[samp_index,]
      y_rf <- ytrain[samp_index]
 
      w_init <- rep(1/length(samp_index),length(samp_index))
      
      tpx_init <- numeric(nrow(xtest))
      
      tpx_rf <- AdaPMT(ms=n_iteration,M=M,depth=depth,trainx=x_rf,trainy=y_rf,xtest=xtest,step=step,tpx_init=tpx_init,w_init=w_init,size=size,lab_list=lab_list)

      rf_test[n,,] <-tpx_rf 

      
    }
    
    
    weights <- hist_accs/sum(hist_accs)
    
    pred_multi<- apply(apply(rf_test,c(1,2),function(block) which.max(block)  ), 2,function(col) lab_list[as.numeric(names(which.max(table(col))))])
    
    return(pred_multi)
    
  }

}

