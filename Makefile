all:
	git ls-files | entr -cr clj -J-Xmx3G -M -m nn2 --report stderr

cuda:
	LD_LIBRARY_PATH=/opt/libcublas-linux-x86_64-11.8.1.74-archive/lib clj -M -m nn3
