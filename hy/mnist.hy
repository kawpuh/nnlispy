(require hyrule [->])
(import torch)
(import torch [nn])
(import torch.utils.data [DataLoader])
(import torchvision [datasets])
(import torchvision.transforms [ToTensor])

(setv train_data
      (datasets.MNIST :root "data"
                      :train True
                      :download True
                      :transform (ToTensor)))

(setv test_data
      (datasets.MNIST :root "data"
                      :train False
                      :download True
                      :transform (ToTensor)))

(setv batch_size 8)

(setv train_dataloader
      (DataLoader train_data :batch_size batch_size))
(setv test_dataloader
      (DataLoader test_data :batch_size batch_size))

;; (for [[X y] test_dataloader]
;;   (print f"shape of x: {X.shape}")
;;   (print f"shape of y: {y.shape} {y.dtype}"))

(setv device (if (torch.cuda.is_available) "cuda" "cpu"))
(print f"using {device}")

(defclass NN [nn.Module]
  (defn __init__ [self]
    (.__init__ (super NN self))
    (setv self.flatten (nn.Flatten))
    (setv self.linear_sigmoid_stack (nn.Sequential
                                      (nn.Linear 784 100)
                                      (nn.Sigmoid)
                                      (nn.Linear 100 10)
                                      (nn.Sigmoid))))
  (defn forward [self x]
    (let [x (self.flatten x)
          logits (self.linear_sigmoid_stack x)]
      logits)))

(setv model (.to (NN) device))
;; (print model)

(setv loss_fn (nn.CrossEntropyLoss))
(setv optim (torch.optim.SGD (model.parameters) :lr 0.2))

(defn train [dataloader model loss_fn optim]
  (setv size (len dataloader.dataset))
  (model.train)
  (for [[batch [X y]] (enumerate dataloader)]
    (setv X (.to X device))
    (setv y (.to y device))
    (setv pred (model X))
    (setv loss (loss_fn pred y))
    (optim.zero_grad)
    (loss.backward)
    (optim.step)
    (when (= 0 (% batch 100))
      (setv loss (.item loss))
      (setv current (* batch (len X)))
      (print f"loss: {loss :>7f} [{current :>5d}/{size :>5d}]"))))

(defn test [dataloader model loss_fn]
  (setv size (len dataloader.dataset))
  (setv num_batches (len dataloader))
  (model.eval)
  (setv test_loss 0)
  (setv correct 0)
  (with [(torch.no_grad)]
    (for [[X y] dataloader]
      (setv X (.to X device))
      (setv y (.to y device))
      (setv pred (model X))
      (setv test_loss (+ test_loss (.item (loss_fn pred y))))
      (setv correct (+ correct (-> (= y (pred.argmax 1))
                                   (.type torch.float)
                                   (.sum)
                                   (.item))))))
  (setv test_loss (/ test_loss num_batches))
  (setv correct (/ correct size))
  (print "Test error:")
  (print f"\taccuracy: {(* 100 correct) :>0.1f}%")
  (print f"\tavg loss: {test_loss :>8f}\n"))

(setv epochs 20)

(for [t (range epochs)]
  (print f"Epoch {(+ 1 t)}\n--------------------")
  (train train_dataloader model loss_fn optim)
  (test test_dataloader model loss_fn))
(print "done")
