library(ComplexHeatmap)
library(circlize)
library(cluster)
library(svglite)
library(png)

mat <- read.table('data/cell_by_topic_embeds.csv', row.names = 1,sep=',',
                  header = TRUE, stringsAsFactors = FALSE,check.names=F)
mat<-t(mat)
metadata  <- read.table('data/cell_by_topic_metadata.csv',sep=',', row.names = 1,
                        header = TRUE, stringsAsFactors = FALSE)
colorscheme = colorRamp2(c(min(mat) ,0, max(mat)), c("blue", "white", "red"))

ann <- data.frame(metadata$condition, metadata$cell_type)
colnames(ann) <- c('condition','cell_type')

colours <- list(
  'condition' = c('type 2 diabetes mellitus' = 'black', 'normal' = 'green'),
  'cell_type' = c('B cell'='red','T cell'='grey',
                  'endothelial cell' = 'blue','epithelial cell of proximal tubule' = 'purple',
                  'fibroblast' = 'pink', 'kidney distal convoluted tubule epithelial cell' = 'yellow',
                  'kidney loop of Henle thick ascending limb epithelial cell' = 'cyan', 
                  'kidney loop of Henle thin ascending limb epithelial cell' = 'limegreen',
                  'kidney proximal convoluted tubule epithelial cell'='wheat',
                  'kidney proximal straight tubule epithelial cell'='violet',
                  'mononuclear cell' = 'maroon','parietal epithelial cell' = 'magenta',
                  'podocyte'='ivory','renal alpha-intercalated cell'='deeppink','renal beta-intercalated cell'='brown',
                  'renal principal cell'='lightseagreen'
  ))

colAnn <- HeatmapAnnotation(df = ann,
                            which = 'column',
                            annotation_width = unit(c(1, 2), 'cm'),
                            gap = unit(0, 'mm'),
                            col = colours,
                            annotation_legend_param = list(
    labels_gp = gpar(fontsize = 7)  
  ))

hmap <- Heatmap(
  mat,
  col=colorscheme,
  column_title = "Kidney Diabetes",
  name='Topic Intensity',
  column_dend_side = "bottom",
  column_dend_height = unit(10, "mm"),
  clustering_method_rows='average',
  show_row_names = TRUE,
  show_column_names = FALSE,
  cluster_rows = TRUE,
  cluster_columns = FALSE,
  show_column_dend = FALSE,
  show_row_dend = TRUE,
  row_dend_reorder = TRUE,
  column_dend_reorder = TRUE,
  clustering_method_columns = "average",
  row_names_gp = gpar(fontsize = 10),
  top_annotation=colAnn,
  )

pdf(file = "output/cell_by_topic.pdf",width=8,height=6)
draw(hmap, heatmap_legend_side="right", annotation_legend_side="right", merge_legend = TRUE)
dev.off()

png("output/cell_by_topic.png")
draw(hmap, heatmap_legend_side="right", annotation_legend_side="right", merge_legend = TRUE)
dev.off()


row_order_hmap = row_order(hmap)
actual_topic_name = dimnames(hmap@matrix)[1]
mylist = list() 
mylist[["order1"]] = row_order_hmap
mylist[["topic"]] = actual_topic_name

# write.table(as.data.frame(mylist),file="order.csv", quote=F,sep=",",row.names=F)
