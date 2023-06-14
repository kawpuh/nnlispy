run:
	clj -J-Xmx3G -M -m simple-gamir --report stderr
dev:
	git ls-files | entr -cr clj -J-Xmx3G -M -m simple-gamir --report stderr
