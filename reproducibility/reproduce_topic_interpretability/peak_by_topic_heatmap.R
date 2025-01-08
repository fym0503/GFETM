library(ComplexHeatmap)
library(circlize)
library(cluster)
library(svglite)

 
# user defined set of topics

mat <- read.table('data/peak_by_topic_panel_transform.csv', row.names = 1,sep=',',check.names = F,
                  header = TRUE, stringsAsFactors = FALSE)
mat <- t(mat)

colorscheme=colorRamp2(c(min(mat),0,max(mat)), c( "blue","white", "red"))

hmap <- Heatmap(
  mat,
  col=colorscheme,
  name='Topic Intensity',
  column_dend_side = "bottom",
  show_row_names = TRUE,
  show_column_names = TRUE,
  cluster_rows = FALSE,
  cluster_columns = FALSE,
  column_names_gp = gpar(fontsize = 10),
  row_names_gp = gpar(fontsize = 15),
  rect_gp = gpar(col = "grey", lwd = 0.1))


pdf(file = "output/peak_by_topic.pdf",width=12,height=4)
draw(hmap, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()


svglite("output/peak_by_topic.svg",width=8,height=3)
draw(hmap, heatmap_legend_side="left", annotation_legend_side="right")
dev.off()