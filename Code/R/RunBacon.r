library(rbacon)

Bacon(core="SmithPond1", thick=5, coredir="",
 prob=0.95, d.min=0, d.max=680, add.bottom=TRUE, d.by=4, seed=NA, depths.file=FALSE, 
 depths=c(), depth.unit="cm", age.unit="yr", unit=depth.unit, acc.shape=1.5, acc.mean=20, 
 mem.strength=10, mem.mean=0.5, suggest=TRUE, accept.suggestions=FALSE, reswarn=c(10,200),
 remember=TRUE, ask=TRUE, run=TRUE, defaults="defaultBacon_settings.txt", ssize=4000, plot.pdf=TRUE)