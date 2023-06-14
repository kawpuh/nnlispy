(ns gamir
  (:require
   [clojure.core.matrix :as m]
   [clojure.core.matrix.random :as rand]
    ; [meta-csv.core :as csv]
    ; [denisovan.core]
    ; [clatrix.core]
    ; [clojure.core.async :as async]
    ; [clojure.java.io :as io]
    ; [tech.v3.datatype.char-input :as ch]
   [clojure.pprint :as pp]))

; (m/set-current-implementation :vectorz)

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
        weights (map
                 (fn [[n m]] (m/scale (rand/sample-normal [n m]) (/ 1 (Math/sqrt m))))
                 (map vector (rest sizes) (butlast sizes)))]
    [weights biases]))

(defn sigmoid [z]
  (->> z
       (-)
       Math/exp
       (+ 1)
       (/ 1)))

(defn sigmoid' [z]
  (* (sigmoid z) (- 1 (sigmoid z))))

; (def cost-fn
;   {:quadratic {:delta (fn [zm av yv] (m/mul (m/sub av yv) (m/emap sigmoid' zm)))}
;    :cross-entropy {:delta (fn [_zm av yv] (m/sub av yv))}})

(defn feedforward [[weights biases] xv] ; => yv
  (reduce
   (fn [av [wm bv]]
     (m/emap sigmoid (m/add bv (m/inner-product wm av))))
   xv
   (map vector weights biases)))

; (defn backprop [[weights biases] [xv yv]] ; => [nabla-w nabla-b]
;   (defn forward [] ; => [zm am]
;     (reduce (fn [[zs as] [wm bv]]
;               (let [zv (m/add bv (m/inner-product wm (last as)))]
;                 [(conj zs zv) (conj as (m/emap sigmoid zv))]))
;             [[] [xv]]
;             (map vector weights biases)))
;   (defn backward [[zm am]] ; => [nabla-w nabla-b]
;     (let [r-zm (reverse zm)
;           r-weights (reverse weights)
;           r-am (reverse am)
;           delta-0 ((get-in cost-fn [:cross-entropy :delta])
;                    (first r-zm)
;                    (first r-am)
;                    yv)]
;       (reduce (fn [[nabla-weights nabla-biases] [zv wm av]]
;                 (let [delta- (first nabla-biases)
;                       delta (m/mul
;                              (m/inner-product (m/transpose wm) delta-)
;                              (m/emap sigmoid' zv))]
;                   [(conj nabla-weights (m/outer-product delta av))
;                    (conj nabla-biases delta)]))
;               ;; initial value
;               [(list (m/outer-product delta-0 (second r-am)))
;                (list delta-0)]
;               ;; reduction coll
;               (map vector
;                    (rest r-zm)
;                    r-weights
;                    (drop 2 r-am)))))
;   (-> (forward)
;       (backward)))

; (defn parameterize-mini-batch-fn [eta mini-batch-size scaled-reg-parameter]
;   (let [nabla-scaling (- (/ eta mini-batch-size))
;         l2-weight-scaling (- 1 (* eta scaled-reg-parameter))]
;     (fn train-mini-batch [[weights biases] training-batch]
;       (let [[nabla-weights nabla-biases]
;             (reduce
;              (fn nabla-sum [[nabla-weights nabla-biases] [dnabla-weights dnabla-biases]]
;                [(pmap m/add nabla-weights dnabla-weights)
;                 (map m/add nabla-biases dnabla-biases)])
;              (pmap (partial backprop [weights biases]) training-batch))]
;         [(pmap m/scale-add weights (repeat l2-weight-scaling)
;                nabla-weights (repeat nabla-scaling))
;          (map m/add-scaled biases nabla-biases (repeat nabla-scaling))]))))

; (defn sgd [[weights biases] training-data {:keys [epochs mini-batch-size eta lambda]}]
;   (defn do-epoch [[weights- biases-]]
;     (time
;      (let [scaled-reg-parameter (/ lambda (count training-data))
;            training-batches (partition mini-batch-size (shuffle training-data))]
;        (reduce
;         (parameterize-mini-batch-fn eta mini-batch-size scaled-reg-parameter)
;         [weights- biases-]
;         training-batches))))
;   (nth (iterate do-epoch [weights biases]) epochs))

(defn index-max [output-vector]
  (first
   (apply max-key second (map-indexed vector output-vector))))

; (defn evaluate [[weights biases] test-data]
;   (mapv
;    (fn [[x y]]
;      (let [res (feedforward [weights biases] x)]
;        [[(index-max res) (m/emax res)] (translate-output y)]))
;    test-data))

; (defn load-mnist-train []
;   (with-open [reader (io/reader "./mnist/mnist_train.csv")]
;     (doall
;      (ch/read-csv-compat reader))))

; (defn load-mnist-test []
;   (with-open [reader (io/reader "./mnist/mnist_test.csv")]
;     (doall
;      (ch/read-csv-compat reader))))

(defn one-hot-encode
  ([hot-i] (one-hot-encode hot-i 9))
  ([hot-i n]
   (for [i (range n)]
     (if (= i hot-i) 1 0))))

; (defn convert-csv [csv]
;   (pmap
;    (fn [row]
;      [(m/array (map (fn [i] (/ (Integer/parseInt i) 255)) (rest row)))
;       (m/array (one-hot-encode (Integer/parseInt (first row))))])
;    csv))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Game Impl
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-board []
  ; [[0 0 0] [0 0 0] [0 0 0]]
  (m/matrix [[0 0 0] [0 0 0] [0 0 0]]))

(defn gameover? [board]
  (some #(or (= % [1 1 1])
             (= % [-1 -1 -1]))
        (concat
          ;; rows
         (for [row board]
           row)
          ;; cols
         (for [i (range 3)]
           (mapv #(get % i) board))
          ;; diagonals
         (list
          (mapv #(get-in board [% %]) (range 3))
          (mapv (fn [i j] (get-in board [i j]))
                (range 3)
                (reverse (range 3)))))))

(defn human-readable-vals [v]
  (case v
    0 " "
    1 "x"
    -1 "o"))

(defn display-as-board [board]
  ;; takes arbitary 3x3
  (->> board
       (map #(interpose "|" %)) ;; add vertical lines in board
       (map (partial apply str))
       ; (interpose "-----") ;; add horizontal lines)
       (interpose "\n")
       (apply str)))

(defn display-board [board]
  ;; takes 3x3 with computer values #{0 1 -1}
  (->> board
       (map (partial map human-readable-vals)) ;; translate
       display-as-board))

(defn valid-move? [board n]
  (and
   (<= 1 n 9)
   (= 0 (nth (apply concat board) (dec n)))))

(defn play-move [board player-turn n]
  (let [n (dec n)
        r (quot n 3)
        c (rem n 3)]
    (assoc-in board [r c] player-turn)))

(defn parse-move [s]
  (try (Integer/parseInt s)
       (catch Exception _ -1)))

(defn human-move [board player-turn]
  (loop []
    (println "Board State:")
    (println (display-board board))
    (println)
    (println board)
    (println)
    (println "Input move for" (human-readable-vals player-turn))
    (println (display-as-board [[1 2 3] [4 5 6] [7 8 9]]))
    (println "Move: ")
    (let [inp (read-line)
          n (parse-move inp)]
      (println)
      (cond
        (not (valid-move? board n)) (do (println "Invalid move") (recur))
        :else (play-move board player-turn n)))))

(defn play-human-game []
  (loop [board (new-board)
         player-turn 1]
    (print "\033\143")
    (if-not (gameover? board)
      (recur (human-move board player-turn) (- player-turn))
      (do
        (println "Game Over")
        (println (display-board board))))))

(defn computer-move [board player-turn i]
  ;; return board and true-reward
  (if (valid-move? board (inc i))
    [0 (play-move board player-turn (inc i))] ;; need to give true reward
    [-1 board] ;; invalid move count as loss
    ))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; value: float
; reward: float
; policy_logits: Dict[Action, float]
; hidden_state: List[float]

(defn dynamics-network []
  ;; takes action, hidden-state
  ;; returns reward, new hidden state
  (network [18 81 10]))

(defn prediction-network []
  ;; takes hidden-state
  ;; returns value, policy-logits
  (network [9 81 10]))

(defn representation []
  ;; takes played-moves
  ;; returns initial hidden-state
  )

(def initial-hidden-state [1 1 1 1 1 1 1 1 1])

(def DISCOUNT 0.9)

(defn play-game [pred-net dyn-net]
  (loop [board (new-board)
         player-turn 1
         actions []]
    (if (gameover? board) [board player-turn actions]
      (let [[v & policy] (feedforward pred-net initial-hidden-state)
            ranked-policy (sort-by second > (map-indexed vector policy))
            action (index-max policy)
            [reward & hidden-state] (feedforward dyn-net (concat (one-hot-encode action)
                                                                 initial-hidden-state))
            [true-reward board'] (computer-move board player-turn action)]
        (recur board' (- player-turn) (conj actions action))))))

(defn print-main []
  (let [pred-net (prediction-network)
        dyn-net (dynamics-network)
        board (new-board)
        player-turn 1
        [v & policy] (feedforward pred-net initial-hidden-state)
        ranked-policy (sort-by second > (map-indexed vector policy))
        action (index-max policy)
        [reward & hidden-state] (feedforward dyn-net (concat (one-hot-encode action)
                                                             initial-hidden-state))
        [true-reward board] (computer-move board player-turn action)]
   (display-board board)))

(defn -main [& args]
  (println (print-main)))

;; need a model

