#!/usr/bin/env Rscript

# This script parses the cohort metadata together with the corresponding FPKM values
#  and it both writes a GSEA compatible, gene-symbol centric expression matrix
#  and outputs the phenotype file required to run the GSEA analysis to the std output.
# To run:
#  	$ cd checkpoint-trials/
#   $ Rscript ./scripts/GSEA_Aggregate_Expression_Data.R analysis/GSEA/expression_fpkm.txt > analysis/GSEA/benefit_phenotype.cls
# and use the files epxression_fpkm.txt and benefit_phenotype.cls in your GSEA analysis.

args <- commandArgs(trailingOnly = TRUE);
outputFile <- args[1];

cohortData <- read.csv("data/melanoma-cohort.csv", header=TRUE, stringsAsFactors=FALSE);
patientCohort <- subset(cohortData, RNA_FILE_ID != "");
rownames(patientCohort) <- patientCohort[, 2];  # index by study id

# extract study id, tissue of origin and biopsy (pre/post) info for patients
metaData.1 <- read.csv("data/s1.csv", header=TRUE, stringsAsFactors=FALSE)[, c(1, 4, 12)];
metaData.2 <- read.csv("data/s2.csv", header=TRUE, stringsAsFactors=FALSE)[, c(1, 4, 12)];
colnames(metaData.1) <- colnames(metaData.2) <- c("id", "tissue", "biopsy");
# combine them and index them with study id
metaData <- rbind(metaData.1, metaData.2);
rownames(metaData) <- metaData[, 1];
# append this information to the patientCohort
patientCohort <- cbind(patientCohort, metaData[row.names(patientCohort), ]);

rnaData <- NULL;
geneFileNameTemplate <- "data/mskcc-melanoma-rna/Sample_%s-cufflinks_output/genes.fpkm_tracking";
sampleNames <- c(NULL);
outcomes <- c(NULL);

for(i in 1:nrow(patientCohort)) {
	# Extract identification
	sample <- patientCohort[i, ];
	rnaId <- sample$RNA_FILE_ID;
	# Locate the expression file
    sampleFile <- sprintf(geneFileNameTemplate, rnaId);
    if(!file.exists(sampleFile)) { next; }  # Skip if the sample doesn't exist
    # Load the data in
    sampleRna <- read.csv(sampleFile, header=T, sep="\t", stringsAsFactors=FALSE);
    benefit <- if(sample$benefit == "TRUE") "benefit" else "no_benefit";
    tissue <- sample$tissue;
    biopsy <- sample$biopsy;
    outcomes <- c(outcomes, paste(tissue, biopsy, benefit, sep="-"));

    # Filter bad quality results
    filteredData <- subset(sampleRna, FPKM_status != "FAIL" & FPKM_conf_lo > 0);
    # Aggregate FPKM values by taking the mean of them
    aggData <- aggregate(x=list(FPKM=filteredData$FPKM), by=list(id=filteredData$gene_short_name), FUN=mean);
    # Now that we have the unique ids let's make them row names for our matrix
    sampleData <- as.matrix(aggData[, "FPKM"]);
    rownames(sampleData) <- aggData[, "id"];
	sampleNames <- c(sampleNames, rnaId);

    # Now combine it with the rest of the sample data
    if(!is.null(rnaData)) {
    	idx <- intersect(rownames(rnaData), rownames(sampleData))
    	rnaData <- cbind(rnaData[idx, ], sampleData[idx, ]);
    	rownames(rnaData) <- idx;
    	colnames(rnaData) <- sampleNames;
    } else {
    	rnaData <- sampleData;
    }
}

# Remove non-informative genes from the dataset
rnaData.devs <- apply(rnaData, MARGIN=1, FUN=sd, na.rm=TRUE);
rnaData <- rnaData[which(rnaData.devs > 0), ];  # Filter out genes for which std dev is 0

# Re-format for GSEA
idColumn <- rownames(rnaData);
gseaData <- cbind(idColumn, idColumn, rnaData);  ## twice the same column intentionally
colnames(gseaData) <- c("NAME", "DESCRIPTION", colnames(rnaData));
write.table(x=gseaData, file=outputFile, sep="\t", quote=FALSE, row.names=FALSE);

# Output the phenotype file
uniqueClasses <- unique(outcomes);
cat(length(sampleNames), length(uniqueClasses), "1\n", sep=" ");
cat("#", uniqueClasses, sep=" ");
cat("\n", sep="");
cat(outcomes, sep=" ");
cat("\n", sep="");
cat("#", sampleNames, "\n", sep=" ");  # This is not required but good for debugging