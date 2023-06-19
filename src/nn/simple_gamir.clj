(ns simple-gamir
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

(defn tanh [z]
  (let [e-2z (Math/exp (* 2 z))]
    (/ (- e-2z 1) (+ e-2z 1))))

(defn tanh' [z]
  (- 1 (Math/pow (tanh z) 2)))

; (def cost-fn
  ; {:quadratic {:delta (fn [zm av yv] (m/mul (m/sub av yv) (m/emap sigmoid' zm)))}
;    :cross-entropy {:delta (fn [_zm av yv] (m/sub av yv))}})

(defn delta-L [zm av yv]
  ;; MSE delta
  (m/mul (m/sub av yv) (m/emap tanh' zm)))

(defn mse [av yv]
  (m/magnitude-squared (m/sub av yv)))

(defn feedforward [[weights biases] xv] ; => yv
  (reduce
   (fn [av [wm bv]]
     (->>
      (m/add bv (m/inner-product wm av))
      (m/emap tanh)))
   xv
   (map vector weights biases)))

(defn backprop [[weights biases] xv yv] ; => [nabla-w nabla-b]
  (defn forward [] ; => [zm am]
    (reduce (fn [[zs as] [wm bv]]
              (let [zv (m/add bv (m/inner-product wm (last as)))]
                [(conj zs zv) (conj as (m/emap tanh zv))]))
            [[] [xv]]
            (map vector weights biases)))
  (defn backward [[zm am]] ; => [nabla-w nabla-b]
    (let [r-zm (reverse zm)
          r-weights (reverse weights)
          r-am (reverse am)
          delta-0 (delta-L (first r-zm) (first r-am) yv)]
      #_(clojure.pprint/pprint [(first r-zm) (first r-am) yv
                                delta-0
                                (mse (first r-am) yv)])
      (reduce (fn [[nabla-weights nabla-biases] [zv wm av]]
                (let [delta- (first nabla-biases)
                      delta (m/mul
                             (m/inner-product (m/transpose wm) delta-)
                             (m/emap tanh' zv))]
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

(def DISCOUNT 0.9)

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

(defn top-n-index [n output-vector]
  (take n (map first (sort-by second > (map-indexed vector output-vector)))))

(defn one-hot-encode
  ([hot-i] (one-hot-encode hot-i 9))
  ([hot-i n]
   (for [i (range n)]
     (if (= i hot-i) 1 0))))

(defn make-targets [reward actions]
  (let [rewards (iterate (partial * DISCOUNT -1) reward)]
    (map cons rewards (map one-hot-encode actions))))

(defn test-backprop-prednet [pred-net reward states actions] ; => [nabla-w nabla-b]
  (map (partial backprop pred-net) states (make-targets reward actions)))

(defn parameterize-mini-batch-fn [eta mini-batch-size scaled-reg-parameter]
  (let [nabla-scaling (- (/ eta mini-batch-size))
        l2-weight-scaling (- 1 (* eta scaled-reg-parameter))]
    (fn train-mini-batch [[weights biases] xvs yvs]
      (let [[nabla-weights nabla-biases]
            (reduce
             (fn nabla-sum [[nabla-weights nabla-biases] [dnabla-weights dnabla-biases]]
               [(pmap m/add nabla-weights dnabla-weights)
                (map m/add nabla-biases dnabla-biases)])
             (map (partial backprop [weights biases]) xvs yvs))]
        [(pmap m/scale-add
               weights (repeat l2-weight-scaling)
               nabla-weights (repeat nabla-scaling))
         (map m/add-scaled biases nabla-biases (repeat nabla-scaling))]))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; Game Impl
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn translate-vals [v]
  (case v
    0 " "
    1 "x"
    -1 "o"))

(defn new-state []
  (m/array [1 0 0 0 0 0 0 0 0 0]))

(defn draw? [board]
  (not (some (partial = 0) board)))

(defn gameover? [[_player-turn & board]]
  ; ((0 1 2) (3 4 5) (6 7 8) (0 3 6) (1 4 7) (2 5 8) (0 4 8) (2 4 6))
  (or
   (draw? board)
   (some #(or (= % [1 1 1])
              (= % [-1 -1 -1]))
         (concat
            ;; rows
          (for [offset (range 0 9 3)]
            (for [i (range 3)]
              (nth board (+ offset i))))
            ;; cols
          (for [offset (range 3)]
            (for [i (range 0 9 3)]
              (nth board (+ offset i))))
            ;; diagonals
          (list
           (map (fn [i j] (nth board (+ i j))) (range 3) (range 0 9 3))
           (map (fn [i j] (nth board (+ i j))) (reverse (range 3)) (range 0 9 3)))))))

(defn display-board [board]
  ;; takes 3x3 with computer values #{0 1 -1}
  (->> board
       (map (partial map translate-vals)) ;; translate
       (map #(interpose "|" %)) ;; add vertical lines in board
       (map (partial apply str))
       ; (interpose "-----") ;; add horizontal lines)
       (interpose "\n")
       (apply str)))

(defn print-state [[player-turn & board]]
  ;; takes 1x9 with computer values #{0 1 -1}
  (println (translate-vals player-turn) "to move")
  (println (display-board (partition 3 board))))

(defn valid-move? [board i]
  (and
   (<= 0 i 8)
   (= 0 (nth board i))))

(defn play-move [[player-turn & _board :as state] i]
  (-> state
      (assoc (inc i) player-turn) ;; inc because 0 is playerturn
      (assoc 0 (- player-turn))))

(defn reward [[player-turn & board]]
  (if (some #(= % [player-turn player-turn player-turn])
            (map (fn [[i j k]] [(nth board i) (nth board j) (nth board k)])
                 [[0 1 2] [3 4 5] [6 7 8] [0 3 6] [1 4 7] [2 5 8] [0 4 8] [2 4 6]]))
    1
    0))

(defn computer-move [[player-turn & board :as state] i]
  ;; return [true-reward state]
  (if (not (valid-move? board i))
    [-1 (concat [(- player-turn)] board)] ;; invalid move count as loss
    (let [state' (play-move state i)]
      [(reward state') state'])))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

; value: float
; reward: float
; policy_logits: Dict[Action, float]
; hidden_state: List[float]

(defn prediction-network []
  ;; takes state [player-turn & board]
  ;; returns value, policy-logits
  (network [10 324 10]))

(defn pred-move [pred-net state]
  (let [[v & policy] (feedforward pred-net (concat state))
        action-candidates (top-n-index 3 policy)]
    (->> (map
          (juxt identity
                (fn [action]
                  (let [[reward p-state] (computer-move state action)]
                    (if (not= 0 reward) reward
                        (first (feedforward pred-net p-state))))))
          action-candidates)
         (apply max-key second)
         first)))

(def K 3)
(def N 3)

(defn self-play [pred-net]
  (loop [true-reward 0
         [[turn & board :as state] & past-states :as states] (list (new-state))
         actions (list)]
    (println true-reward actions)
    (if (or (not= 0 true-reward) (draw? board)) [true-reward past-states actions]
        (let [action (pred-move pred-net state)
              [true-reward state'] (computer-move state action)]
          (recur true-reward (conj states state') (conj actions action))))))

; {:epochs 60
;  :mini-batch-size 10
;  :eta 0.2
;  :lambda 5.0}

(defn train-until-length []
  (loop [net (prediction-network)
         i 0]
    (let [[reward states actions] (self-play net)
          ; res (test-backprop-prednet net reward states actions)
          mini-batch-fn (parameterize-mini-batch-fn 0.0008 1 1.0)
          trained-net (mini-batch-fn net states (make-targets reward actions))
          game-length (count actions)]
      (println actions)
      (println game-length i)
      (recur trained-net (inc i))

      #_(if (> game-length 3)
          trained-net
          (recur trained-net (inc i))))))

(defn -main [& args]
  ; (train-until-length)
  (self-play (prediction-network))
  ; (println (pred-move (prediction-network) (new-state)))
  (println "done")
  #_(let [net (prediction-network)
          [reward states actions] (self-play net)
          res (backprop-prednet net reward states actions)
          mini-batch-fn (parameterize-mini-batch-fn 0.2 1 1.0)
          trained-net (mini-batch-fn net states (make-targets reward actions))]
      (println "done")
      (clojure.pprint/pprint @write-atom)
    ; (println (m/shape res))
    ; (println "actions:" actions)
    ; (println "result:" reward)
      #_(doseq [state states]
          (print-state state))))

;; need a model

