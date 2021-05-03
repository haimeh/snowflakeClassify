library(imager)

valueOverwrite=function(rawData,imDim,batch.size)
{
  colorChan <- 1
  dim(rawData) <- c(imDim,batch.size)
   

  rotateIndex <- sample(c(T,F,F),batch.size,replace = T)
  rawData[,,,rotateIndex] <- permute_axes(rawData[,,,rotateIndex, drop=F],"yxzc")
  data <- rawData
  

  #for(i in 1:2)
  #{
  #  resizeIndex <- sample(c(T,F,F,F),batch.size,replace = T)

  #  if(any(resizeIndex)) {
  #  	cX <- runif(1,.45,.55)
  #  	cY <- runif(1,.45,.55)
  #  	sDX <- ceiling(runif(1,.75,1.25)*imDim[1])
  #  	sDY <- ceiling(runif(1,.75,1.25)*imDim[2])

  #  	rawDataRes <- resize(rawData[,,,resizeIndex,drop=F], size_x=sDX, size_y=sDY, 
  #  	    			    interpolation_type=0, boundary_conditions=1, 
  #  	    			    centering_x=cX, centering_y=cY)
  #  	rawData[,,,resizeIndex] <- resize(rawDataRes, size_x=imDim[1], size_y=imDim[2], 
  #  	    			    interpolation_type=sample(c(1,5),1))
  #  }
  #}



  #data <- rawData-min(rawData)
  #data <- data/max(data)
  #dim(data) <- c(imDim,batch.size)
  #
  #noise <- NULL
  #for(i in seq_len(colorChan*batch.size))
  #{
  #  noise <- append(noise,rnorm(n=prod(imDim), mean=runif(1,-.15,.15), sd=runif(1,0,.1)))
  #}
  #dim(noise) <- c(imDim,batch.size)
  #noise <- as.array(isoblur(as.cimg(noise),runif(1,0,2)))
  #
  #data <- data+noise
  #rm(noise)
  
  #if(sample(c(T,F),1))
  #{
  #  data <- as.array(data-min(data))
  #  data <- data/max(data)
  #}else{
  #  data[data<0]<-0
  #  data[data>1]<-1
  #}
   
  dim(data) <- c(imDim,batch.size)
  
  flipX <-   sample(c(T,F,F),batch.size,replace = T)
  flipY <-   sample(c(T,F,F),batch.size,replace = T)
  
  data[,,,flipX] <- mirror(data[,,,flipX,drop=F],axis="x")
  data[,,,flipY] <- mirror(data[,,,flipY,drop=F],axis="y")
  
  gc()
  #data[is.na(data) | is.nan(data) | is.infinite(data)]<-(.5)
  
  return(data)
}
