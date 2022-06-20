all:
	echo "src/nn.clj" | entr -cr clj -M --report stderr src/nn.clj
