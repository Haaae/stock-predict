from src.Domain.Train_Model.Modeler import get_model

def model_train(loss_function,
                optimizer,
                patience,
                seq_len, 
                x_train,
                y_train,
                batch_size,
                epoch,
                validation_split,
                tensorboard_callback):

    model, early_stopping = get_model(loss_function, optimizer, patience, seq_len)

    # Model traing
    # 3단계: 모델 학습
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epoch,
        validation_split=validation_split,
        callbacks=[tensorboard_callback],
        verbose=0)
    
    return model