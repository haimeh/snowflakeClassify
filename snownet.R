library("mxnet")
library("imager")
library("e1071")
source("measuringScript.R")

is.mx.dataiter <- function(x) {
	  any(is(x, "Rcpp_MXNativeDataIter") || is(x, "Rcpp_MXArrayDataIter"))
}

snowIter <- setRefClass("snowIter",
	fields=c("data",
		"iter",
		"data.shape"),
	contains = "Rcpp_MXArrayDataIter",
	
	methods=list(
		initialize=function(iter=NULL,
						data,
						data.shape){
		data_len <- prod(data.shape)
		print(paste0("shp:",data.shape))
		array_iter <- mx.io.arrayiter(data,
									label=rep(0,ncol(data)),
									batch.size=min(ncol(data),128))
		.self$iter <- array_iter
		.self$data.shape <- data.shape
		.self
	  },
	  
	  value=function(){
		val.x <- as.array(.self$iter$value()$data)
		val.x[is.na(val.x) | is.nan(val.x) | is.infinite(val.x)] <- .5
		dim(val.x) <- c(.self$data.shape,.self$data.shape,1,ncol(val.x))
		
		list(data=mx.nd.array(val.x))
	},
	  
	iter.next=function(){
		.self$iter$iter.next()
	},
	reset=function(){
		.self$iter$reset()
	},
	num.pad=function(){
		.self$iter$num.pad()
		},
	finalize=function(){
		.self$iter$finalize()
		}
	)
)




mx.simple.bind <- function(symbol, ctx, dtype ,grad.req = "null", fixed.param = NULL, slist, ...) {
		if (!mxnet:::is.MXSymbol(symbol)) stop("symbol need to be MXSymbol")
	
		if (is.null(slist)) {
			stop("Need more shape information to decide the shapes of arguments")
		}
	
	arg.arrays <- sapply(slist$arg.shapes, function(shape) {
		mx.nd.array(array(0,shape), ctx)
	}, simplify = FALSE, USE.NAMES = TRUE)
	aux.arrays <- sapply(slist$aux.shapes, function(shape) {
		mx.nd.array(array(0,shape), ctx)
	}, simplify = FALSE, USE.NAMES = TRUE)

	grad.reqs <- lapply(names(slist$arg.shapes), function(nm) {
		if (nm %in% fixed.param) {
			print("found fixed.param")
			"null"
		}else if (!endsWith(nm, "label") && !endsWith(nm, "data")) {
			grad.req
		}else {
			"null"
		}
	})
	print("BOUND")
	return(mxnet:::mx.symbol.bind(symbol, ctx,
					 arg.arrays=arg.arrays,
					 aux.arrays=aux.arrays,
					 grad.reqs = grad.reqs))
}

mx.model.init.params <- function (symbol, input.shape, fixed.shape, 
								  output.shape, initializer, ctx){
	if (!mxnet:::is.MXSymbol(symbol)){
		stop("symbol needs to be MXSymbol")
	}
	arg_lst <- list(symbol = symbol)
	arg_lst <- append(arg_lst, input.shape)
	arg_lst <- append(arg_lst, output.shape)
	arg_lst <- append(arg_lst, fixed.shape)

	slist <- do.call(mx.symbol.infer.shape, arg_lst)
	if (is.null(slist)){stop("Not enough information to get shapes")}

	arg.params <- mx.init.create(initializer, slist$arg.shapes,
		ctx, skip.unknown = TRUE)
	aux.params <- mx.init.create(initializer, slist$aux.shapes,
		ctx, skip.unknown = FALSE)
	return(list(arg.params = arg.params, aux.params = aux.params, slist))
}

predict.MXFeedForwardModel_cust <- function (model, X, ctx = NULL, array.batch.size = 128, 
											 array.layout = "auto",allow.extra.params = FALSE){
	if (is.array(X) || is.matrix(X)) {
		if (array.layout == "auto") {
			array.layout <- mxnet:::mx.model.select.layout.predict(X,model)
		}
		if (array.layout == "rowmajor") {
			X <- t(X)
		}
	}
	X <- mxnet:::mx.model.init.iter(X, NULL, batch.size = array.batch.size,
		is.train = FALSE)
	X$reset()
	if (!X$iter.next())
		stop("Cannot predict on empty iterator")
	dlist = X$value()
	# extract shape based on symbol name
	## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	namedShapes <- lapply(model$symbol$arguments,function(x){
				  as.integer(strsplit(gsub("\\(([^()]*)\\)|.", "\\1", x, perl=T),",")[[1]])
	})

	names(namedShapes) <- model$symbol$arguments
	fixed.shapes <- namedShapes[sapply(namedShapes,function(x)length(x)!=0)]
	names(arg.shapes) <- names(model$arg.params)
	#aux.shapes <- lapply(model$aux.params,dim)
	#names(aux.shapes) <- names(model$aux.params)
	fixed.shapes[names(arg.shapes)] <- arg.shapes
	## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

	input.names <- names(dlist)
	input.shape <- sapply(input.names, function(n){dim(dlist[[n]])}, simplify = FALSE)
	input.shape <- input.shape[1]
	#fixed.shapes <- append(namedShapes,lapply(fixed.param,dim))
	initialized <- mx.model.init.params(symbol=model$symbol, 
				input.shape=input.shape, 
				fixed.shape=fixed.shapes, 
				output.shape=NULL, 
				initializer=mx.init.uniform(0), 
				ctx=ctx)
	params <- initialized[1:2]
	slist <- initialized[[3]]
	arg_lst <- list(symbol = model$symbol, ctx = ctx, data = dim(dlist$data),
		grad.req = "null", slist=slist)

	pexec <- do.call(mx.simple.bind, arg_lst)
	if (allow.extra.params) {
		model$arg.params[!names(model$arg.params) %in% arguments(model$symbol)] <- NULL
	}
	mx.exec.update.arg.arrays(pexec, model$arg.params, match.name = TRUE)
	mx.exec.update.aux.arrays(pexec, model$aux.params, match.name = TRUE)
	packer <- mxnet:::mx.nd.arraypacker()
	X$reset()
	while (X$iter.next()) {
		dlist = X$value()
		mx.exec.update.arg.arrays(pexec, list(data = dlist$data),
			match.name = TRUE)
		mx.exec.forward(pexec, is.train = FALSE)
		out.pred <- mx.nd.copyto(pexec$ref.outputs[[1]], mx.cpu())
		padded <- X$num.pad()
		oshape <- dim(out.pred)
		ndim <- length(oshape)
		packer$push(mxnet:::mx.nd.slice(out.pred, 0, oshape[[ndim]] - padded))
	}
	X$reset()
	return(packer$get())
}





#sortedCropsToSVM <- function(baseDir){
embedSnowflakes <- function(baseDir){
	imgFolders <- list.files(baseDir,recursive=F, full.names=T)
	embeddingsDF <- NULL
	measurements <- list()
	classVec <- NULL
	imgLocation <- NULL

	#index <- !(mainImgName %in% cropedImgNames)
	#index <- rep(1,length(list.files(baseDir,recursive=T,full.names=F)))
	index <- 0
	howmanyimgs <- length(list.files(baseDir,recursive=T,full.names=F))
	withProgress(message = 'Embedding', value = 0,{
		progressTicker <- 0
	for(folder in imgFolders){
		imgVec75 <- NULL
		imgVec175 <- NULL
		imgVec275 <- NULL
		print(folder)
		imgNames <- list.files(folder,recursive=F, full.names=T)
		for(imgName in imgNames){try({
			print(imgName)
			index <- index+1

			classVec <- append(classVec, basename(folder))
			imgLocation <- append(imgLocation, imgName)
			img <- load.image(imgName)
			if(dim(img)[4]>1) img <- B(img)
			
			#measurement <- try({getSnowflakeStats(img)*1})
			#if (class(measurement)=="try-error"){
			#	measurements[[index]] <- c(NA, NA, NA)
			#}else{
			#	measurements[[index]] <- measurement
			#}
			img = resize_halfXY(img)

			measurement <- try({getSnowflakeStats(img)})
			if (class(measurement)=="try-error"){
					#getSnowflakeStats(img,T)
				measurements[[index]] <- c(NA, NA, NA)
			}else{
				measurement <- measurement * c(2,2,4)
				measurements[[index]] <- measurement
			}

			edges = c(as.numeric(img[1,,,]),as.numeric(img[,1,,]),as.numeric(img[width(img),,,]),as.numeric(img[,height(img),,]))
			newValues = rnorm(n=length(edges),mean=mean(edges),sd=sd(edges)*.25)
			img[1,,,] <- sample(newValues,height(img))
			img[width(img),,,] <- sample(newValues,height(img))
			img[,1,,] <- sample(newValues,width(img))
			img[,height(img),,] <- sample(newValues,width(img))
			halfImg <- img
	
			if(max(dim(img))<=75){
				resizedImg <- as.numeric(resize(halfImg, size_x=75, size_y=75, 
												interpolation_type=0,boundary_conditions=1, centering_x=.5,centering_y=.5))
				imgVec75 <- cbind(imgVec75, resizedImg)
			}else if(max(dim(img))<=175){
				resizedImg <- as.numeric(resize(halfImg, size_x=175, size_y=175, 
												interpolation_type=0,boundary_conditions=1, centering_x=.5,centering_y=.5))
				imgVec175 <- cbind(imgVec175, resizedImg)
			}else{
				resizedImg <- as.numeric(resize(halfImg, size_x=275, size_y=275, 
												interpolation_type=0,boundary_conditions=1, centering_x=.5,centering_y=.5))
				imgVec275 <- cbind(imgVec275, resizedImg)
			}
		})}
		
		if(length(imgVec75) > 0){
			dataIter <- snowIter$new(data = imgVec75, data.shape = 75)
			netEmbedding <- t(predict.MXFeedForwardModel_cust(nnModel,
														dataIter,
														array.layout = "colmajor",
														ctx= mx.gpu(),
														allow.extra.params=T))
			embeddingsDF <- rbind(embeddingsDF,netEmbedding)
			rm(dataIter)
			gc()
		}
		if(length(imgVec175) > 0){
			dataIter <- snowIter$new(data = imgVec175, data.shape = 175)
			netEmbedding <- t(predict.MXFeedForwardModel_cust(nnModel,
														dataIter,
														array.layout = "colmajor",
														ctx= mx.gpu(),
														allow.extra.params=T))
			embeddingsDF <- rbind(embeddingsDF,netEmbedding)
			rm(dataIter)
			gc()
		}
		if(length(imgVec275) > 0){
			dataIter <- snowIter$new(data = imgVec275, data.shape = 275)
			netEmbedding <- t(predict.MXFeedForwardModel_cust(nnModel,
														dataIter,
														array.layout = "colmajor",
														ctx= mx.gpu(),
														allow.extra.params=T))
			embeddingsDF <- rbind(embeddingsDF,netEmbedding)
			rm(dataIter)
			gc()
		}
		#incProgress(1/sum(index), detail = paste(basename(imgName)," -- ",progressTicker,"of",sum(index)))
		incProgress(length(imgNames), detail = paste(basename(imgName)," -- ",progressTicker,"of",howmanyimgs))
	}
	})

	return(list(pathData=imgLocation, classData=classVec, hashData=embeddingsDF, measurements = do.call(rbind, measurements)))

	#counts <- table(classVec)
	#classWeights <- counts/sum(counts)
	#svmModel <- svm( as.factor(classVec)~., data=t(embeddingsDF) , type="C-classification",  kernel = "polynomial", degree=2, cost = 100, scale = FALSE, class.weights=classWeights, inverse=T, probability=T)
	#return(svmModel)
}


