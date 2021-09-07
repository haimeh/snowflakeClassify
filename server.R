library("shiny")
library("DT")
library("imager")
library("mxnet")
library("e1071")
library("MASS")
source("crystalCropp.R")
source("snownet.R")

#options(browser="/usr/bin/firefox")

#appScripts <- system.file("shiny_app", package="crystalCrystaldR")
#sapply(list.files(path=appScripts,pattern="*_serverside.R",full.names = T),source,.GlobalEnv)

#networks <- system.file("extdata", package="crystalCrystaldR")
#mxnetModel <- mxnet::mx.model.load(file.path(networks,'crystal_triplet32_4096_crystalal'), 5600)

#model <- mx.model.load("SWA_snowflakeTestTripRefine2",0000)
#model <- mx.model.load("SWA_sf64TripRefine",0000)
#model <- mx.model.load("SWA_sf64TripF",0000)
model <- mx.model.load("harmonic64",0000)
embeddingMean <- c(
-1.78670534,   1.61472011,  -1.30197684, -10.19537330,  -2.37761307,
 1.87344104,  -2.21611227,  -2.67849631,  -0.59680935,  -1.48055685,
-1.47556361,  -2.92614473,  -1.67263501,   5.53355529,   1.73912589,
 0.08333041
)

myInternals = internals(model$symbol)
anchor_symbol = myInternals[[match("anchor_output", outputs(myInternals))]]

nnModel <<- list(symbol = anchor_symbol,
			arg.params = model$arg.params,
			aux.params = model$aux.params)
class(nnModel) <- "MXFeedForwardModel"
nnModel <<- nnModel

load("svmModels.Rdata")
svmModels_reactive <- reactiveValues(svmModels=svmModels)





#options(browser="/usr/bin/firefox")

plotCrystalTrace <- function(crystal){
	if(length(crystal)>0){
		par(mar = c(0,0,0,0))
		plot(crystal, ann=FALSE, axes = FALSE)
	}else{
		print("not an image")
	}
}


predictClasses <- function(svmModel,embeddingsDF){
	res <- predict(svmModel, newdata=embeddingsDF, probability=T)
	prob <- attr(res,"probabilities")
	top3 <- t(apply(attr(res,"probabilities"),1,function(x){ attr(res,"levels")[order(x,decreasing=T)[1:3]] }))
	colnames(top3) <- c("1st", "2nd", "3rd")
	return(list(top3=top3,Probabilities=prob))
}

generateNewSVM <- function(classVec, embeddingsDF){
	counts <- table(classVec)
	classWeights <- counts/sum(counts)
	svmModel <- svm( as.factor(classVec)~., data=embeddingsDF , type="C-classification",  kernel = "polynomial", 
					 degree=2, cost = 100, scale = FALSE, class.weights=classWeights, inverse=T, probability=T)
	return(svmModel)
}

fileRename <- function(from, to) {
	todir <- dirname(to)
	if (!isTRUE(file.info(todir)$isdir)) dir.create(todir, recursive=TRUE)
	file.rename(from = from,  to = to)
}

sortImagesByClass <- function(classifications, imgPaths, newRootDir){
	newImgPaths <- imgPaths
	i <- 0
	for(imgPath in imgPaths){
		i = i+1
		newImgPaths[i] <- file.path(newRootDir,classifications,basename(imgPath))
		fileRename(from = imgPath, to = newImgPaths[i])
	}
	return(newImgPaths)
}

# --- Server Logic -----------------------------------------------------------------------------------------
# ==================================================================================================================

function(input, output, session) {
	
	# --- stop r from app ui
	session$onSessionEnded(function(){
	  stopApp()
	})
	
	sessionStorage <- new.env()
	
	
	
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#<-><-><-><-> Rank Table <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	sessionQuery <- reactiveValues(hashData=NULL,
								measurements=NULL,
								pathData=NULL,
								classData=NULL) #0
	rankTable <- reactiveValues(Name=NULL,
								Distance=NULL,
								Probabilities=NULL,
								editCount=0)
	dendrogramMem <- reactiveValues(hmData=NULL)

	# --- get data into r 
	observeEvent(input$embedQuery,{
		if(!is.null(input$queryDirectory)){
			# generate embeddings 
			dirNames <- unique(sapply(list.files(input$queryDirectory, recursive=T, full.names=T,pattern="*.jpg$|*.JPG$|*.png$|*.PNG$"),function(x){(dirname(x))}))
			for(subdir in dirNames){
				print(subdir)
				embeddedData <- embedSnowflakes(subdir)
				sessionQuery$hashData <- rbind(sessionQuery$hashData, embeddedData$hashData)
				sessionQuery$measurements <- rbind(sessionQuery$measurements, embeddedData$measurements)
				sessionQuery$pathData <- append(sessionQuery$pathData, embeddedData$pathData)
				sessionQuery$classData <- append(sessionQuery$classData, embeddedData$classData)

				rankTable$editCount <- rankTable$editCount+1
				print("finished subdir")
			}
			print("finished all embed")
		}
	})

	output$selectedSVM_UI <- renderUI({ selectInput("selectedSVM", "Select SVM model:", choices = names(svmModels_reactive$svmModels)) })

	observeEvent(input$classify,{
		if(!is.null(sessionQuery$hashData)){
			predictionData <- predictClasses(svmModels[[input$selectedSVM]], sessionQuery$hashData)
			classes <- cbind(basename(sessionQuery$pathData),predictionData[["top3"]], sessionQuery$measurements)
			rankTable$Distance <- classes #rbind(classes,rankTable$Distance)
			rankTable$Probabilities <- predictionData$Probabilities
			#measurements <- do.call(rbind,sessionQuery$measurements)
			if(input$moveClassifications){
				#classifications, imgPaths, newRootDir
				#BUG: fix the source of classesPathdata
				sessionQuery$pathData <- sortImagesByClass(classifications=predictionData[["top3"]][,1], 
								  imgPaths=sessionQuery$pathData, 
								  newRootDir=input$sortedDirectory)
			}
			rankTable$editCount <- rankTable$editCount+1
		}
	})
	# --- clear session memory
	observeEvent(input$clearSVM,{
		if(input$selectedSVM != "Default"){
			svmModels_reactive$svmModels[[input$selectedSVM]] <- NULL
			print(paste("delete",input$selectedSVM))
			print(paste("models",names(svmModels_reactive$svmModels)))
			save(svmModels_reactive$svmModels,file="svmModels.Rdata")
		}
	})
	observeEvent(input$clearEmbed,{
			rankTable$Distance <- NULL
			rankTable$editCount <- rankTable$editCount+1
			sessionQuery$hashData <- NULL
			sessionQuery$measurements <- NULL
			sessionQuery$pathData <- NULL
			sessionQuery$classData <- NULL
	})
	# ----
	observeEvent(input$trainSvm,{
		if(!is.null(sessionQuery$hashData)){
				browser()
			svmModels_reactive$svmModels[[input$svmName]] <- generateNewSVM(classVec=sessionQuery$classData, embeddingsDF=sessionQuery$hashData)
			save(svmModels_reactive$svmModels,file="svmModels.Rdata")
		}
	})
	
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#<-><-> Classification Display <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	# --- rank table downloads -----------------------------------------
	#NameTable
	#IDTable
	#DistanceTable
	#DT::dataTableOutput("matchDistance"),downloadButton("DistanceTableDownload")
	output$DistanceTableDownload <- downloadHandler(
		filename = function() {
			paste0("ClassTable_",gsub(" ", "_", date()),".csv")
		},
		content = function(file) {
			#index <- seq_len(min(as.integer(input$rankLim),ncol(rankTable$Distance)))
			write.csv(
				#rankTable$Distance[,index],file, row.names = T
				cbind(sessionQuery$pathData, round(rankTable$Probabilities,3), sessionQuery$measurements), file, row.names = F
			)
		}
	)
	
	# --- table of ranked matches -----------------------------------------
	
	# --- rankTable rendering
	tableOptions = list(lengthChange = T, 
					rownames=F, 
					ordering=F, 
					paging = T,
					scrollY = "500px",
					scrollX = "750px",
					pageLength = 1000, lengthMenu = list('500', '1000','2000', '10000'))
	output$matchDistance <- DT::renderDataTable({
		index <- seq_len(min(as.integer(input$rankLim),ncol(rankTable$Distance)))
		#round(rankTable$Distance[,index, drop=FALSE],2)
		rankTable$Distance[,index, drop=FALSE]
		},
		selection = list(mode="single",target = "row"),
		options = tableOptions
	)
	
	## --- crystal image
	observeEvent(c(input$matchDistance_cell_clicked),{
		output$imageTableQuery <- renderPlot({
			if(!is.null(input$matchDistance_rows_selected)){
				plotCrystalTrace(load.image(sessionQuery$pathData[input$matchDistance_rows_selected]))
			}else{
				NULL
			}
		})
	})

	
	
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	#<-><-> Hierarchical Clustering <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
	#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
	observeEvent(input$clickPointSet,{
		#1.002 : -.033
		if(is.numeric(input$clickPointSet$y)){
			print(paste(input$clickPointSet$x, input$clickPointSet$y))
			correctedPercent <- (input$clickPointSet$y+.033)/(1.002 + .033)
			correctedPercent <- abs(1-pmax( 0, pmin( correctedPercent, 1)))

			estIndex <- round(correctedPercent*length(dendrogramMem$hmData$rowInd))
			print(estIndex)
			if(estIndex<1)estIndex=1
			
			output$imageTableQuery <- renderPlot({
				plotCrystalTrace(load.image(sessionQuery$pathData[dendrogramMem$hmData$rowInd[estIndex]]))
			})
		}
	})
	# --- Hash Reference Table
	observeEvent(c(input$embedQuery, rankTable$editCount), {
						 #browser()
						 print(length(sessionQuery$hashData))
						 print(nrow(sessionQuery$hashData))
		if((length(sessionQuery$hashData))>1){
			#sessionStorage$permutation <- NULL
			
			hashData <- sessionQuery$hashData-embeddingMean
			
			#dist_mat <- dist(hashData, method = 'euclidean')
			#hclust_avg <- hclust(dist_mat, method = 'ward.D')
			#sessionStorage$permutation <- hclust_avg$order
			
			
			#testHashTable <- round(hashData[sessionStorage$permutation,],-1)/10
			#testHashTable <- round(hashData[sessionStorage$permutation,])
			
			#tblBreaks <- quantile(testHashTable, probs = seq(.05, .95, .05), na.rm = TRUE)
			#tblColors <- round(seq(255, 40, length.out = length(tblBreaks) + 1), 0) %>% {paste0("rgb(",.,",",.,",255)")}
			
			#colnames(testHashTable) <- c(letters[1:16])

			#heatmap(hash, Colv=NA, hclustfun=function(x){hclust(dist(x,method="euclidean"),method="ward.D")})
			output$hashComparison <- renderPlot({
					dendrogramMem$hmData <- heatmap(hashData, Colv=NA, hclustfun=function(x){hclust(dist(x,method="euclidean"),method="ward.D")},
					margins = c(0,0),
					labRow = NA, labCol = NA, col=hcl.colors(128,"viridis", rev = FALSE))
			})
		}
	})
	
	
	##############################################################################################
	#<-><-> Crop <-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><-><->
	##############################################################################################
	
	#cropPath <- normalizePath(input$queryDirectory,"/")
	#dir.create(file.path(paste0(cropPath,"_snowflake-Crops")), showWarnings = FALSE)

			  
	observeEvent(input$cropRawImages,{
		print("cropping")
		cropPath <- normalizePath(input$queryDirectory,"/")
		newDir <- normalizePath(input$cropDirectory,"/")
		#newDir <- file.path(paste0(cropPath,"_crystalCrystaldR-Crops"))
		dir.create(newDir, showWarnings = FALSE)
		print(cropPath)
		print(newDir)
		
		cropDirectory(searchDirectory=cropPath,
						saveDirectory=newDir,
						minXY=45,
						maxXY=700,
						includeSubDir=T,
						mimicDirStructure=T)
	})
}

#runApp()
