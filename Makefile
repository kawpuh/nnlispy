all:
	git ls-files | entr -cr clj -J-Xmx3G -M -m nn2 --report stderr
