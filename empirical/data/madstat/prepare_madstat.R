raw <- "C:/Users/X.Zhu58/Documents/hypernew/final/final/empirical/data/madstat"
load(file.path(raw, "AuthorPaperInfo.RData"))

paper_ids <- sort(unique(AuPapMat$idxPap))
authors <- split(AuPapMat$idxAu, AuPapMat$idxPap)
authors <- lapply(authors[as.character(paper_ids)], sort)
years <- AuPapMat$year[match(paper_ids, AuPapMat$idxPap)]

write_txt <- function(x, name) {
  connection <- file(file.path(raw, name), "wb")
  writeLines(as.character(x), connection)
  close(connection)
}

write_txt(lengths(authors), "madstat-nverts.txt")
write_txt(unlist(authors, use.names = FALSE), "madstat-simplices.txt")
write_txt(as.integer(years), "madstat-times.txt")
write_txt(paper_ids, "madstat-paper-ids.txt")
cat("Exported", length(paper_ids), "papers and", sum(lengths(authors)), "author-paper records.\n")
