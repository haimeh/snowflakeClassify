
library(shiny)

fluidPage(
	titlePanel(title=div(
		"Atmospheric Crystal",
		img(src='crystal.png', height = '65px',align = "right")
		
	)),
	sidebarLayout(
		sidebarPanel(
			#%%%%%%%%%%%%%%%%%%%%%%%%
			# save stuff
			#%%%%%%%%%%%%%%%%%%%%%%%%
			actionButton("clearEmbed","Clear Embeddings"),
			h1(" "),#just for space
			
			#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# QUERY INPUT
			#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# all images in directory
			conditionalPanel(
				condition = "input.inputType !== 'Classify'",
				textInput(
					inputId = "queryDirectory",
					label = "Source Directory"
				)
			),
	
			radioButtons(
				inputId = "inputType",
				label = "Select Source Directory Action",
				choices = c("Crop/Cut","Embed","Classify","Train"),
				inline = T,
				selected = "Embed"
			),
	
			
			#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			# image vs rdata input
			#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
			conditionalPanel(
				condition = "input.inputType == 'Crop/Cut'",
				actionButton(
					inputId = "cropRawImages",
					label = "Crop Images"
				),
				h1(" "),
				textInput(
					inputId = "cropDirectory",
					label = "New directory for cropped images"
				),
				h1(" "),
				paste0("Crop/Cut large images and save each set of sub images inside the specified crop directory. 
						Each image will generate its own directory with the cropped contents. The original image is not modified.")
			),
			conditionalPanel(
				condition = "input.inputType == 'Classify'",
				h1(" "),
				uiOutput("selectedSVM_UI"),
				#selectInput("selectedSVM", "Select SVM model:", choices = names(svmModels))
				fluidRow(
					column(width = 6, class = "well",
						actionButton(
							inputId = "classify",label = "Classify Images"
						),
					),
					column(width = 6, class = "well",
						actionButton("clearSVM","Delete Selected SVM")
					)
				),
				h1(" "),
				checkboxInput(inputId="moveClassifications",
							  label = "Move images according to classification.",
							  value=F),
				conditionalPanel(
				condition="input.moveClassifications",
				textInput(
					inputId = "sortedDirectory",
					label = "Directory for sorted and classified images"
				)),
				h1(" "),
				paste0(" Classify cropped images and if desired, move the images into new directories based on the result of their classification.")
			),
			conditionalPanel(
				condition = "input.inputType == 'Train'",
				h1(" "),
				textInput(
					inputId = "svmName",
					label = "Name for new SVM",
					value = "Custom"
				),
				h1(" "),
				actionButton(
					inputId = "trainSvm",
					label = "Train New SVM"
				),
				h1(" "),
				paste0("Train a new SVM classifier based on the directory structure inside the source directory.")
			),
			conditionalPanel(
				condition = "input.inputType == 'Embed'",
				actionButton(
					inputId = "embedQuery",
					label = "Generate Embeddings"
				),
				h1(" "),
				paste0("Embed cropped images to a lower dimmensional representation for the svm to classify. 
					Each image embedding is associated with the name of the directory in which it is saved.")
			)
		),
	   
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
		# MAIN PANEL
		# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	
		mainPanel(
		tags$style(type="text/css",
					  ".shiny-output-error { visibility: hidden; }",
					  ".shiny-output-error:before { visibility: hidden; }"),
			fluidRow(
				column(width = 6, class = "well",
					uiOutput("headerTableQuery"),
					plotOutput("imageTableQuery")#,click = clickOpts(id = "clickPointSet",clip = TRUE))
				)
			),
			tabsetPanel(id = "mainTblPanel",
				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Matches tab - display ordered table of query to reference matches
				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				tabPanel("Classification",
					fluidRow(
						DT::dataTableOutput("matchDistance"),downloadButton("DistanceTableDownload")
					)
				),
				
				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				# Clusters tab - view dendrogram, representing matches as heirarchical cluster
				#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
				tabPanel("Clusters",
					#uiOutput("displayWindows"),
					column(width = 12,
						#DT::dataTableOutput("hashComparison",width = '1600px')# '1824px' '1280px'
						plotOutput("hashComparison",click = clickOpts(id = "clickPointSet",clip = F), 
								   width = '1000px',height = '970px')
					)
				)
			)
		)
	)
)
