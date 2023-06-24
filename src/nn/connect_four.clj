(ns connect-four
  (:require
   [clojure.core.matrix :as m]
   [clojure.core.matrix.random :as rand]
   [clojure.string]
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

; (defn index-max [output-vector]
;   (first
;    (apply max-key second (map-indexed vector output-vector))))

; (defn top-n-index [n output-vector]
;   (take n (map first (sort-by second > (map-indexed vector output-vector)))))

; (defn one-hot-encode
;   ([hot-i] (one-hot-encode hot-i 9))
;   ([hot-i n]
;    (for [i (range n)]
;      (if (= i hot-i) 1 0))))

;(defn make-targets [reward actions]
;  (let [rewards (iterate (partial * DISCOUNT -1) reward)]
;    (map cons rewards (map one-hot-encode actions))))

;(defn test-backprop-prednet [pred-net reward states actions] ; => [nabla-w nabla-b]
;  (map (partial backprop pred-net) states (make-targets reward actions)))

;(defn parameterize-mini-batch-fn [eta mini-batch-size scaled-reg-parameter]
;  (let [nabla-scaling (- (/ eta mini-batch-size))
;        l2-weight-scaling (- 1 (* eta scaled-reg-parameter))]
;    (fn train-mini-batch [[weights biases] xvs yvs]
;      (let [[nabla-weights nabla-biases]
;            (reduce
;             (fn nabla-sum [[nabla-weights nabla-biases] [dnabla-weights dnabla-biases]]
;               [(pmap m/add nabla-weights dnabla-weights)
;                (map m/add nabla-biases dnabla-biases)])
;             (pmap (partial backprop [weights biases]) xvs yvs))]
;        [(pmap m/scale-add
;               weights (repeat l2-weight-scaling)
;               nabla-weights (repeat nabla-scaling))
;         (map m/add-scaled biases nabla-biases (repeat nabla-scaling))]))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;; Game Impl
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(defn new-state []
  ;; turn, board (6x7 left to right starting from bottom row)
  (m/array (concat [1] (repeat 42 0))))

(defn draw? [board]
  (not (some (partial = 0) board)))

(def directions #{:n :ne :e :se :s :sw :w :nw})

(defn unchecked-move-direction [i dir]
  (case dir
    :n (+ i 7)
    :ne (+ i 8)
    :e (+ i 1)
    :se (- i 8)
    :s (- i 7)
    :sw (- i 6)
    :w (- i 1)
    :nw (+ i 6)))

(defn move-direction [i dir]
  ;; returns nil if the new position is invalid
  (let [i' (unchecked-move-direction i dir)]
    (if (<= 0 i' 41) i' nil)))

(defn display-tile [tile]
  (case tile
    0 \-
    1 \X
    -1 \O))

(defn display-board [board]
  (->> board
       (map display-tile)
       (partition 7)
       (map (partial clojure.string/join " "))
       reverse
       (clojure.string/join "\n")))

(defn- internal-play-move [[player-turn & board :as state] i]
  ;; return altered-i, altered-state
  (loop [check-i i]
    (cond
      (< 41 check-i) [nil state] ;; invalid move

      (= 0 (nth board check-i)) ;; valid move
      [check-i
       (-> state
           (update 0 -)
           (assoc (inc check-i) player-turn))]

      :else (recur (+ check-i 7)))))

(defn parse-move [s]
  (try (Integer/parseInt s)
       (catch Exception _ nil)))

(defn connect-four? [[now-player-turn & board] altered-i]
  ;; was the previous move a winning move?
  (let [prev-player-turn (- now-player-turn)]
    (loop [[dir & rem-directions] directions]
      (cond
        (nil? dir) false

        (every? (comp (partial = prev-player-turn) #(nth board % nil))
                (take 4 (iterate #(unchecked-move-direction % dir) altered-i)))
        true

        :else (recur rem-directions)))))

(defn human-move []
  (let [input (read-line)
        parsed-input (parse-move input)]
    (if (some? parsed-input) parsed-input (recur))))

(defn human-game []
  (loop [[player-turn & board :as state] (new-state)]
    (println (display-board board))
    (when-not (draw? state)
      (println "0 1 2 3 4 5 6")
      (println (display-tile player-turn) "to move")
      (let [[altered-i state'] (internal-play-move state (human-move))]
        (if (connect-four? state' altered-i)
          (do
            (println (display-board (rest state')))
            (println (display-tile player-turn) "wins!"))
          (recur state'))))))

;(defn valid-move? [board i]
;  (and
;   (<= 0 i 8)
;   (= 0 (nth board i))))

;(defn play-move [[player-turn & _board :as state] i]
;  (-> state
;      (assoc (inc i) player-turn) ;; inc because 0 is playerturn
;      (assoc 0 (- player-turn))))

;(defn reward [[_ & board] player-turn]
;  ;; most of the time
;  (if (some #(= % [player-turn player-turn player-turn])
;            (map (fn [[i j k]] [(nth board i) (nth board j) (nth board k)])
;                 [[0 1 2] [3 4 5] [6 7 8] [0 3 6] [1 4 7] [2 5 8] [0 4 8] [2 4 6]]))
;    1
;    0))

;(defn computer-move [[player-turn & board :as state] i]
;  ;; return [true-reward state]
;  (if (not (valid-move? board i))
;    [-1 (concat [(- player-turn)] board)] ;; invalid move count as loss
;    (let [state' (play-move state i)]
;      [(reward state' player-turn) state'])))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; value: float
;; reward: float
;; policy_logits: Dict[Action, float]
;; hidden_state: List[float]

;(defn prediction-network []
;  ;; takes state [player-turn & board]
;  ;; returns value, policy-logits
;  (network [10 324 10]))

;(defn pred-move [pred-net state]
;  (let [[v & policy] (feedforward pred-net (concat state))
;        action-candidates (top-n-index 6 policy)]
;    (->> (map
;          (juxt identity
;                (fn [action]
;                  (let [[reward p-state] (computer-move state action)]
;                    (if (not= 0 reward) reward
;                        (first (feedforward pred-net p-state))))))
;          action-candidates)
;         (apply max-key second)
;         first)))

;(def K 3)
;(def N 3)

;(defn self-play [pred-net]
;  (loop [true-reward 0
;         [[turn & board :as state] & past-states :as states] (list (new-state))
;         actions (list)]
;    (if (or (not= 0 true-reward) (draw? board)) [true-reward past-states actions]
;        (let [action (pred-move pred-net state)
;              [true-reward state'] (computer-move state action)]
;          (recur true-reward (conj states state') (conj actions action))))))

;(defn train-until-length []
;  (loop [net (prediction-network)
;         i 0]
;    (let [[reward states actions] (self-play net)
;          ; res (test-backprop-prednet net reward states actions)
;          mini-batch-fn (parameterize-mini-batch-fn 0.0008 1 1.0)
;          trained-net (mini-batch-fn net states (make-targets reward actions))
;          game-length (count actions)]
;      (println reward actions game-length i)
;      (recur trained-net (inc i))

;      #_(if (> game-length 3)
;          trained-net
;          (recur trained-net (inc i))))))

(defn -main [& args]
  ; (let [state (reduce play-move (new-state) [1 2 1 2 1 2 3 3 1])]
  ;   (println (display-board (rest state)))
  ;   (println (gameover? state)))
  (human-game)
  (println "done")
  #_(let [net (prediction-network)
          [reward states actions] (self-play net)
          res (backprop-prednet net reward states actions)
          mini-batch-fn (parameterize-mini-batch-fn 0.2 1 1.0)
          trained-net (mini-batch-fn net states (make-targets reward actions))]
      (clojure.pprint/pprint @write-atom)
    ; (println (m/shape res))
    ; (println "actions:" actions)
    ; (println "result:" reward)
      #_(doseq [state states]
          (print-state state))))

;; need a model

