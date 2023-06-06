(ns nn3
  (:import
    [jcuda Pointer Sizeof]
    [jcuda.jcublas JCublas]
    ))

(defn -main [& args]
  (let [A (float-array [1 2 3 4 5 6 7 8 9])
        pt-A (Pointer.)
        C (float-array 9)]
    (. JCublas cublasInit)
    (. JCublas cublasAlloc 9 (. Sizeof -FLOAT) pt-A)
    (. JCublas cublasSetVector 9 (. Sizeof -FLOAT) (. Pointer to A) 1 pt-A 1)
    (. JCublas cublasGetVector 9 (. Sizeof -FLOAT) pt-A 1 (. Pointer to C) 1)
    (. JCublas cublasFree pt-A)
    (. JCublas cublasShutdown)
    (println (vec C))
    (println A)
    )
  (println "Done"))
