
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
List WoProbitBoost(DataFrame x_train,NumericVector y_train,NumericVector Wo,int M_max=50,double tol=1e-05) {
  
  
  
  NumericVector lab_list= sort_unique(y_train); // number of labels
  
  int n_Class  = lab_list.length() ;    //      # number of different classes
  
  //initialization
  // y_train= as.numeric(y_train!=lab_list[1]) # change class labels to boolean
  
  
  int             n_feature                 = x_train.length() ;
  int             n_obs                     = y_train.length();
  NumericVector   w                         = rep(double (1/n_obs),n_obs);
  //Wo=Wo/sum(Wo);
  CharacterVector feature_name  = x_train.names();
  
  //Wo             = (Wo/sum(Wo)); //#ifelse((is.null(Wo)),rep(1/n_obs,n_obs),Wo) #rep(1/n_obs,n_obs)
  
  NumericVector  f(n_obs);
  NumericVector  p(n_obs,0.5);//p(n_obs);
  NumericVector  coef_intercept(n_feature);
  NumericVector  coef_slope(n_feature);
  DataFrame      boost_feat    =DataFrame::create( Named("feature") = feature_name , _["coef_intercept"] = coef_intercept,_["coef_slope"]=coef_slope );
  
  
  NumericVector llh = {-2*sum((y_train*log(p)+(1-y_train)*log(1-p))*Wo)}; // initialize log-likelihood function
    
    double error=10000.0;
    
    int m= 0;  // number of iteration
    NumericVector z(n_obs);
    NumericVector G(n_obs);
    
    NumericMatrix fitted_ys(n_obs,n_feature) ;//# record loglikelihood of each fitted univariate model
    NumericMatrix temp_beta(2,n_feature) ;
    NumericVector increas_cand_llh(n_feature); //select best attribute
    NumericVector intercept;
    NumericVector slope;
    NumericVector fitted_value;
    int best_feature_index;
    while( error>tol & m < M_max & sum(is_na(z))==0 ){ //exclude the case when there are NAs in z
      
      G = -(2*y_train-1)*f*pnorm((2*y_train-1)*f)-dnorm(f);
      z = (pnorm(f)-y_train)* pow(pnorm((2*y_train-1)*f) ,2) / (G*pnorm(f)*pnorm(-f)+1e-100);  //working response +(1e-20)*f*pow(pnorm((2*y_train-1)*f) ,2) /(G*dnorm(f)+1e-100)
      w = -Wo*dnorm(f)*G / (pow(pnorm((2*y_train-1)*f),2)+1e-100);  //# weight for each sample
      w = w/sum(w) ;            //# normalization
      
      for(int j=0;j<n_feature;j++){
        
        NumericVector x =x_train[j];
        
        double beta_1= sum( w*(x-sum(w*x))* (z-sum(w*z)) ) / (sum(w*pow(x-sum(w*x),2))+1e-100 );
        double beta_0= sum(w*z)-beta_1* sum(w*x);  //sum( (x-mean(x))* (z-mean(z)) ) / sum(pow(x-mean(x),2)) ;
        NumericMatrix::Column  colt = temp_beta(_,j); // reference of j th column of temp_beta
        colt = NumericVector::create(beta_0,beta_1);
        NumericMatrix::Column  colf = fitted_ys(_,j); // reference of j th column of fitted_value
        intercept  = rep(beta_0,n_obs);
        slope      = beta_1 * x ;
        fitted_value= intercept+slope;
        colf = fitted_value;
        increas_cand_llh[j]= mean(w*pow(fitted_value-z,2));
        
        
      }
      best_feature_index = which_min(increas_cand_llh) ;
      
      NumericVector  coef_intercept = boost_feat[1];
      NumericVector   coef_slope = boost_feat[2];

      coef_intercept[best_feature_index]+= temp_beta(0,best_feature_index);//row1[best_feature_index]; //update intercept
      coef_slope[best_feature_index]+= temp_beta(1,best_feature_index);///row2[best_feature_index]; //update coefficient
      
      //Rcout << boost_feat;
      //Rf_PrintValue(temp_beta);
      f += fitted_ys(_,best_feature_index);
      
      p = pnorm(f);
      
      llh.push_back(-2*sum((y_train*log(p+1e-24)+(1-y_train)*log(1-p+1e-24))*Wo)); // #sum(y_train*log(p)+(1-y_train)*log(1-p))
      
      m +=1;
      error = abs(llh[m]-llh[m-1]);
      
    }
    

  List results =List::create(Named("boost_feat") = boost_feat , _["llh"] = llh,_["lab_list"]=lab_list,_["fitted_p"]=p);;
  return(results);
  
}


// [[Rcpp::export]]
NumericVector predictProbit(NumericMatrix x_train,List tree, CharacterVector terminal_nodes) {
  
  int     n_obs  = x_train.nrow();
  NumericVector predicted(n_obs);
  List Tree = tree["pbt"];
  
  List  cur_boost_feats;
  NumericVector coef_intercept;
  NumericVector coef_slope;
  List Tree_cur_node;
  String current_node;
  for(int i=0;i<n_obs;i++){
    NumericVector  cur_obs = x_train(i,_);
    String current_node = terminal_nodes[i];  // extract terminal node
    List Tree_cur_node = Tree[current_node];  // extract corresponding tree
    
    bool fitted_cur = Tree_cur_node["unique"]; // extract corresponding fitted value
    if(fitted_cur){
      
      predicted[i]=  Tree_cur_node["predicted"];
    }else{
      
      cur_boost_feats=Tree_cur_node["boost_feats"];
      
      coef_intercept= cur_boost_feats["coef_intercept"];
      coef_slope = cur_boost_feats["coef_slope"];
      // coef_slope   <- tree[[pred_nodes[i]]]$boost_feats$coef_slope
      double  linear_part = sum(coef_slope*cur_obs)+sum(coef_intercept);
      predicted[i] = R::pnorm(linear_part,0.0,1.0,1,0);// Rcpp::pnorm(linear_part);
      
    }
    
    
  }
  // List fit = Tree["n3"];
  // String current_node = terminal_nodes[1];
  //  List Tree_cur_node = Tree[current_node];
  return(predicted);
}

