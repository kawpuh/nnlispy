all:
	git ls-files | entr -cr clj -J-Xmx3G -M --report stderr src/nn.clj
