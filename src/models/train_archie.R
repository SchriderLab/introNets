library(data.table)
train <- data.frame(fread("/pine/scr/d/d/ddray/ArchIE_glm_features_train_v2.txt")) ## read in the training data

LABEL_COL=209

lr <- train[,1:LABEL_COL] # cut down the extra columns -- 209 is the label (0=not archaic, 1=archaic)
model <- glm(V209 ~ .,family=binomial(link='logit'),data=lr) # train the model

cleanModel2 = function(cm) {
  cm$y = c()
  cm$model = c()

  cm$residuals = c()
  cm$fitted.values = c()
  cm$effects = c()
  cm$qr$qr = c()
  cm$linear.predictors = c()
  cm$weights = c()
  cm$prior.weights = c()
  cm$data = c()
  cm
}

model <- cleanModel2(model)

print(model)

save(model, file = "/pine/scr/d/d/ddray/trained_model_ArchIE_200.Rdata") # save the trained model so we don't have to train it again. can load it with load("trained_model.Rdata")