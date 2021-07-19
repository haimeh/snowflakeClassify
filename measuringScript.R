library("imager")
library("MASS")

radiusInit <- function(shape){
	index <- expand.grid(x=0:(shape[1]-1),y=0:(shape[2]-1))
	samples <- t(t(index)-round((shape-1)/2))

	r <- sqrt(samples[,"x"]^2 + samples[,"y"]^2)
	lim <- max(shape)/7
	r[r<lim] <- lim
	r <- r/max(shape)

	#r <- sqrt(samples[,"x"]^2 + samples[,"y"]^2)
	#r <- r * (r/(1+abs(r/10))) / max(shape)
	
	r <- array(r,dim=shape)
	return(r)
}

rawMoment <- function(img, iord, jord){
	imgDims = dim(img)[1:2]
	nrows = imgDims[1] 
	ncols = imgDims[2]
	x_y = expand.grid(x=1:nrows, y=1:ncols)
	x = x_y$x
	y = x_y$y
	img = img * x^iord * y^jord
	return(sum(img))
}

intertialAxis <- function(img){
	imgSum = sum(img)
	m10 = rawMoment(img, 1, 0)
	m01 = rawMoment(img, 0, 1)
	xMu = m10 / imgSum
	yMu = m01 / imgSum
	u11 = (rawMoment(img, 1, 1) - xMu * m01) / imgSum
	u20 = (rawMoment(img, 2, 0) - xMu * m10) / imgSum
	u02 = (rawMoment(img, 0, 2) - yMu * m01) / imgSum
	cov = array(c(u20, u11, u11, u02), c(2,2))
	return(list(mu=c(x=xMu, y=yMu), cov=cov))
}


makeLines <- function(eigvals, eigvecs, mean, i){
	std = sqrt(eigvals[i])
	vec = 2 * std * eigvecs[,i] / sqrt(sum(eigvecs[,i]^2))
	return(vec)
}

##abs((x2-x1)(y1-y0) - (x1-x0)(y2-y1)) / sqrt((x2-x1)^2 + (y2-y1)^2)
#distanceToLine <- function(coords,p1,p2) {
#	abs((p2$x-p1$x)*(p1$y-coords$y) - (p1$x-coords$x)*(p2$y-p1$y)) / sqrt((p2$x-p1$x)^2 + (p2$y-p1$y)^2)
#}

#imgg <- load.image("/home/jaimerilian/Work/snowflake/app/img6.png")
#imgg <- load.image("/home/jaimerilian/Work/snowflake/app/img8.png")
getSnowflakeStats <- function(imgg,debug=F){
	if(debug){browser()}

	if(max(dim(imgg))<25){
		return(c(majorAxis=NA, minorAxis=NA, area=NA))
	}
	if(debug){browser()}

	qu <- quantile(imgg,seq(0,1,.05))[c(2,20)]
	imgg <- (imgg-qu[1])/qu[2]
	imgg[imgg>1] <- 1
	imgg[imgg<0] <- 0

	
	estMask <- dilate_square((imgg<.15 | imgg>.95),5)
	imgDF <- as.data.frame(imgg)
	bgDF <- imgDF[as.logical(!estMask),]
	linTrend <- lm.ridge(value ~ x+y, data=bgDF, lambda=1)
	trendPred <- array(as.matrix(cbind(const=1,imgDF[,c("x","y")])) %*% coef(linTrend),dim(imgg))
	imgg <- imgg-trendPred
	
	weightBg <- as.cimg(radiusInit(dim(imgg)[1:2])) * as.numeric(!estMask)
	weightFg <- as.cimg(max(weightBg)-radiusInit(dim(imgg)[1:2])) * as.numeric(estMask)
	
	bgmu <- sum(weightBg*imgg)/(sum(weightBg)+1)
	bgsd <- (sum(weightBg*abs(imgg-bgmu))/sum(weightBg)) +.01
	
	fgmu <- sum(weightFg*imgg)/(sum(weightFg)+1)
	fgsd <- (sum(weightFg*abs(imgg-fgmu))/sum(weightFg)) +.01
	
	
	x <- seq(0, 1, length=100)
	bgx <- dnorm(x,bgmu,bgsd) * sum(!estMask)/prod(dim(estMask))
	fgx <- dnorm(x,fgmu,fgsd) * sum(estMask)/prod(dim(estMask))
	
	
	if(sum(bgx>fgx)<3){
		bgRange <- x[1:3]
	}else{
		bgRange <- x[bgx>fgx]
	}
	threshImg <- imgg>(bgRange[length(bgRange)]) | imgg<(bgRange[1])
	
	
	threshImg <- clean(threshImg,3)
	threshImg <- fill(threshImg,7)
	threshImg[c(1,2),,,] <- F
	threshImg[,c(1,2),,] <- F
	threshImg[c(width(imgg),width(imgg)-1),,,] <- F
	threshImg[,c(height(imgg),height(imgg)-1),,] <- F
	imgBlobs <- label(threshImg)
	blobSizes <- table(imgBlobs)
	
	bgBlob <- imgBlobs==0
	fgBlob <- as.pixset(!erode_square(bgBlob,3))
	bgBlob <- as.pixset(dilate_square(bgBlob,5))
	
	bgMu <- mean(imgg[bgBlob])
	bgSd <- sd(imgg[bgBlob])+.01
	
	fgMu <- mean(imgg[fgBlob])
	fgSd <- sd(imgg[fgBlob])+.01
	
	
	x <- seq(0, 1, length=100)
	denom <- length(imgg)
	bgNum <- sum(bgBlob)
	fgNum <- sum(!bgBlob)
	
	
	#if(debug){browser()}


	bgx <- dnorm(x,bgMu,bgSd) * bgNum/denom
	fgx <- dnorm(x,fgMu,fgSd) * fgNum/denom
	#bgRange <- x[bgx>fgx]
	if(sum(bgx>fgx)<3){
		bgRange <- x[1:3]
	}else{
		bgRange <- x[bgx>fgx]
	}
	smoothimg <- imgg
	threshImg <- as.pixset(smoothimg>(bgRange[length(bgRange)]) | smoothimg<(bgRange[1]))
	
	
	threshImg <- clean(threshImg,3)
	threshImg <- fill(threshImg,15)
	threshImg[c(1,2),,,] <- F
	threshImg[,c(1,2),,] <- F
	threshImg[c(width(imgg),width(imgg)-1),,,] <- F
	threshImg[,c(height(imgg),height(imgg)-1),,] <- F
	#browser()
	imgBlobs <- label(threshImg)
	blobSizes <- table(imgBlobs)
	
	mainBlob <- imgBlobs==which.max(blobSizes[-1])
	
	targetBlob <- !(imgBlobs==0)
	targetBlobBlobs <- label(targetBlob)
	
	keepers <- unique(mainBlob * targetBlobBlobs)
	targetBlob[!(targetBlobBlobs %in% keepers)] <- F
	
	
	outline <- dilate_square(targetBlob,3)-targetBlob
	
	
	smoothimg[c(1:5),,,] <- rnorm(height(imgg)*5,bgMu,bgSd/2)
	smoothimg[,c(1:5),,] <- rnorm(width(imgg)*5,bgMu,bgSd/2)
	smoothimg[c(width(imgg):(width(imgg)-4)),,,] <- rnorm(height(imgg)*5,bgMu,bgSd/1)
	smoothimg[,c(height(imgg):(height(imgg)-4)),,] <- rnorm(width(imgg)*5,bgMu,bgSd/1)
	smoothimg <- isoblur(smoothimg,2)
	
	distImg <- isoblur(abs(smoothimg-bgMu),3)
	distImg <- distImg/max(distImg)
	distImg <- resize(distImg,size_x=dim(distImg)[1]*2, size_y=dim(distImg)[2]*2, interpolation_type=0, boundary_conditions=1, centering_x=.5, centering_y=.5)
	outlineBig <- resize(outline,size_x=dim(outline)[1]*2, size_y=dim(outline)[2]*2, interpolation_type=0, boundary_conditions=1, centering_x=.5, centering_y=.5)
	threshImgBig <- resize(threshImg,size_x=dim(outline)[1]*2, size_y=dim(outline)[2]*2, interpolation_type=0, boundary_conditions=1, centering_x=.5, centering_y=.5)
	
	
	meh = isoblur(threshImgBig,15)
	imstat <- intertialAxis(meh+distImg^2)
	eigenstuff <- eigen(imstat$cov)
	
	linestuff1 <- makeLines(eigenstuff$values,eigenstuff$vectors,imstat$mu,1)
	linestuff2 <- makeLines(eigenstuff$values,eigenstuff$vectors,imstat$mu,2)
	
	
	axisCoords <- rbind(imstat$mu-linestuff1,imstat$mu+linestuff1,imstat$mu+linestuff2,imstat$mu-linestuff2)
	
	
	coordinates <- get.locations(outlineBig,function(x)as.logical(x))[,1:2]
	
	#browser()
	#distanceToLine(coordinates,newCoords)
	
	
	edgeDistances <- lapply(1:nrow(axisCoords), function(i){sqrt( colSums((t(coordinates) - axisCoords[i,])^2))})
	names(edgeDistances) <- c("major_0","major_1","minor_0","minor_1")
	
	newCoords <- rbind(
		coordinates[which.max(edgeDistances[["major_0"]] - edgeDistances[["major_1"]]),],
		coordinates[which.max(edgeDistances[["major_1"]] - edgeDistances[["major_0"]]),],
		coordinates[which.max(edgeDistances[["minor_0"]]-1.5*edgeDistances[["minor_1"]]),],
		coordinates[which.max(edgeDistances[["minor_1"]]-1.5*edgeDistances[["minor_0"]]),]
	)
	
	majorDistance <- round(sqrt(sum((newCoords[1,]-newCoords[2,])^2)))
	minorDistance <- round(sqrt(sum((newCoords[3,]-newCoords[4,])^2)))
	
	outlineBigUnmarked <- outlineBig
	for(i in 1:nrow(newCoords)){
		yy = newCoords[i,"y"]
		xx = newCoords[i,"x"]
		#print(paste(xx,yy))
		#distImg[(xx-2):(xx+2),(yy-2):(yy+2),,] <- 2
		outlineBig[(xx-4):(xx+4),(yy-4):(yy+4),,] <- 1
	}
	outlineBig[(imstat$mu["x"]-4):(imstat$mu["x"]+4),(imstat$mu["y"]-4):(imstat$mu["y"]+4),,] <- 1



	newCoords <- rbind(
		coordinates[which.max(edgeDistances[["major_0"]] - edgeDistances[["major_1"]]),],
		coordinates[which.max(edgeDistances[["major_1"]] - edgeDistances[["major_0"]]),],
		coordinates[which.max(edgeDistances[["minor_0"]]-1.5*edgeDistances[["minor_1"]]),],
		coordinates[which.max(edgeDistances[["minor_1"]]-1.5*edgeDistances[["minor_0"]]),]
	)

	#imgReady <- cimg(array(c(outlineBig,outlineBigUnmarked,outlineBigUnmarked),c(dim(outlineBig)[1:2],1,3) ))
	#imgReady <- resize(imgReady,size_x=dim(outline)[1]+8, size_y=dim(outline)[2]+8, interpolation_type=0, boundary_conditions=1, centering_x=.5, centering_y=.5)
	#return(imgReady)

	#return(list(majorAxis=majorDistance,minorAxis=minorDistance, coords=round(newCoords-dim(outline)/2) ))
	return(c(majorAxis=majorDistance, minorAxis=minorDistance, area=sum(targetBlob)))
}
