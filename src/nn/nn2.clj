(ns nn2
  (:require [clojure.core.matrix :as m]
            [clojure.core.matrix.random :as rand]
            ; [meta-csv.core :as csv]
            ; [denisovan.core]
            ; [clatrix.core]
            [clojure.java.io :as io]
            [tech.v3.datatype.char-input :as ch]
            [clojure.pprint :as pp]))

(m/set-current-implementation :vectorz)

;; somethings to keep in mind:
;; weights is a vector of matrices, one for each layer

;; generally things are named by the following convention
;; irregular (non-ndarray) collections are denoted by a word
;; ndarrays are denoted by 2 letter pairs such that
;; first letter describe the contents
;; second letter describes the type:
;; t - tensor, m - matrix, v - vector

(defn network [sizes]
  (let [biases (map rand/sample-normal (rest sizes))
        weights (map (fn [[n m]] (m/scale (rand/sample-normal [n m]) (/ 1 m)))
                            (map vector (rest sizes) (butlast sizes)))]
    [weights biases]))

(defn coerce-num [n]
  (cond
    (= n ##-Inf) Float/MIN_VALUE
    (= n ##Inf) Float/MAX_VALUE
    (Float/isNaN n) 0
    :else n))

(defn sigmoid [z]
  (->> z
       (-)
       Math/exp
       (+ 1)
       (/ 1)))

(defn sigmoid' [z]
  (* (sigmoid z) (- 1 (sigmoid z))))

(def cost-fn
  {:quadratic {:delta (fn [zm av yv] (m/mul (m/sub av yv) (m/emap sigmoid' zm)))}
   :cross-entropy {:delta (fn [_zm av yv] (m/sub av yv))}})

(defn feedforward [[weights biases] xv] ; => yv
  (reduce
   (fn [av [wm bv]]
     (m/emap sigmoid (m/add bv (m/inner-product wm av))))
   xv
   (map vector weights biases)))

(defn backprop [[weights biases] xv yv] ; => [nabla-w nabla-b]
  (defn forward [] ; => [zm am]
    (reduce (fn [[zs as] [wm bv]]
              (let [zv (m/add bv (m/inner-product wm (last as)))]
                [(conj zs zv) (conj as (m/emap sigmoid zv))]))
            [[] [xv]]
            (map vector weights biases)))
  (defn backward [[zm am]] ; => [nabla-w nabla-b]
    (let [r-zm (reverse zm)
          r-weights (reverse weights)
          r-am (reverse am)
          delta-0 ((get-in cost-fn [:cross-entropy :delta])
                   (first r-zm)
                   (first r-am)
                   yv)]
      (reduce (fn [[nabla-weights nabla-biases] [zv wm av]]
                (let [delta- (first nabla-biases)
                      delta (m/mul
                             (m/inner-product (m/transpose wm) delta-)
                             (m/emap sigmoid' zv))]
                  [(conj nabla-weights (m/outer-product delta av))
                   (conj nabla-biases delta)]))
              ;; initial value
              [(list (m/outer-product delta-0 (second r-am)))
               (list delta-0)]
              ;; reduction coll
              (map vector
                   (rest r-zm)
                   r-weights
                   (drop 2 r-am)))))
  (-> (forward)
      (backward)))

(defn train-mini-batch [scaling-factor net training-batch]
  (reduce
   (fn [[weights biases] [x y]]
     (let [[nabla-weights nabla-biases] (backprop [weights biases] x y)]
       [(map m/add-scaled weights nabla-weights (repeat scaling-factor))
        (map m/add-scaled biases nabla-biases (repeat scaling-factor))]))
   net
   training-batch))

(defn sgd [[weights biases] training-data epochs mini-batch-size eta]
  (defn do-epoch [[weights- biases-]]
    (time
     (let [training-batches (partition mini-batch-size (shuffle training-data))
           scaling-factor (- (/ eta mini-batch-size))]
       (reduce
        (partial train-mini-batch scaling-factor)
        [weights- biases-]
        training-batches))))
  (nth (iterate do-epoch [weights biases]) epochs))

(defn translate-output [output-vector]
  (first
   (apply max-key second (map-indexed vector output-vector))))

(defn evaluate [[weights biases] test-data]
  (mapv
    (fn [[x y]]
      (let [res (feedforward [weights biases] x)]
        [[(translate-output res) (m/emax res)] (translate-output y)]))
    test-data))

(defn load-mnist-train []
  (with-open [reader (io/reader "./mnist/mnist_train.csv")]
    (doall
     (ch/read-csv-compat reader))))

(defn load-mnist-test []
  (with-open [reader (io/reader "./mnist/mnist_test.csv")]
    (doall
     (ch/read-csv-compat reader))))

(defn one-hot-encode [n]
  (assert (<= 0 n 9))
  (for [i (range 10)]
    (if (= i n) 1 0)))

(defn convert-csv [csv]
  (map
   (fn [row]
     [(m/array (map (fn [i] (/ (Integer/parseInt i) 255)) (rest row)))
      (m/array (one-hot-encode (Integer/parseInt (first row))))])
   csv))

(def train-data (-> (load-mnist-train)
                    (rest)
                    (convert-csv)))

(def test-data (-> (load-mnist-test)
                   (rest)
                   (convert-csv)))

(println "starting training")
(def trained-net (-> (network [784 100 10])
                     (sgd train-data 10 10 0.5)
                     (vec)))
(println "done")

(let [results (evaluate trained-net test-data)
      tally (->> results
                 (map (fn [[[guess _activation] answer]] (= guess answer)))
                 (frequencies))]
  (pp/pprint tally))
