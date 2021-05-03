library("mxnet")
source("snowflakeDataAug.R")

CustomArrIter <- setRefClass("CustomArrIter",
                                fields=c("myiter","label", "label.shape", "labelTable", "labelBucketIndex", "data", "data.shape", 
					 "batch.size", "dtype", "ctx"),
                                contains = "Rcpp_MXArrayDataIter",
                                methods=list(
                                    initialize=function(myiter=0, data, label, labelTable=list(), labelBucketIndex=list(),  data.shape, batch.size, dtype, ctx){
					.self$data <- data
					.self$label <- label
                                        .self$batch.size <- batch.size
					.self$data.shape <- data.shape
					labelTable = table(label)
					.self$labelTable <- labelTable[labelTable>1]

					.self$labelBucketIndex <- lapply(as.integer(names(.self$labelTable)), function(x) which(.self$label==x))
					names(.self$labelBucketIndex) <- names(.self$labelTable)


					.self$dtype <- dtype
					.self$ctx <- ctx
					.self$myiter <- 0

                                        .self
                                    },
                                    value=function(){
					    .self$myiter <- .self$myiter + 1
				        randomDraw = sample(x=.self$labelTable, size=ceiling(batch.size/2), replace=T)
					sampleDraws = ifelse(as.integer(randomDraw)>=4,4,as.integer(randomDraw))


					randomDrawIndex <- 1
					sampleIndex <- list()
				    	while(length(sampleIndex) <= batch.size){
						try({
						sampleIndex = append(sampleIndex,
								sample(.self$labelBucketIndex[[ names(randomDraw[randomDrawIndex]) ]],
								       sampleDraws[randomDrawIndex] ))
						})
						randomDrawIndex <- randomDrawIndex+1
					}
					if(length(sampleIndex)<batch.size)browser()
					extras <- sample(which(sampleDraws[1:randomDrawIndex]>2),length(sampleIndex)-batch.size)
					sampleIndex <- as.integer(sampleIndex[-extras])

                                        val.x <- .self$data[,,,sampleIndex]
                                        val.x <- val.x/255
					val.y = .self$label[sampleIndex]
					dim(val.y) <- c(1,length(sampleIndex))

					shiftedData <- val.x

					#dim(shiftedData) <- c(data.shape, batch.size)
					shiftedData <- valueOverwrite(rawData=val.x, imDim=.self$data.shape,batch.size=.self$batch.size)

                                        mdata <- mx.nd.array(shiftedData)
					mlabel <- mx.nd.array(val.y)
					return(list(data=mdata, label=mlabel))

                                    },
                                    iter.next=function(){
					   if( .self$myiter == 128){
				    		return(F)
					   }else{
						return(T)
					   }
					    
                                    },
                                    reset=function(){
					    	.self$myiter <- 0
                                    },
                                    num.pad=function(){
					    0
                                    },
                                    finalize=function(){
                                    }
                                )
                            )
