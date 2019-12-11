module_url = '/data/universla_text_embedding/tf_hub'

# Create a graph.
g = tf.Graph()

with g.as_default():
    text_input = tf.placeholder(dtype=tf.string, shape=[None])
    embed = hub.Module(module_url)
    embedded_text = embed(text_input)
    init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])

g.finalize()

texts = ... # The input texts that we want to convert to embeddings

with tf.Session(graph=g) as sess:
    sess.run(init_op)
    embeddings = sess.run(embedded_text, feed_dict={text_input: texts})
    with open('embeddings.txt', 'w') as f:   
        # Write the embeddings to a text file.
        for embedding in np.array(embeddings).tolist():
            s = ','.join([str(e) for e in embedding])
            s += '\n'
            f.write()

# Build a classifier.

X = ... # Read the embeddings text file above.
y = ... # The class label for each text

target = 'Repair_Inspection'

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

with tf.Session(graph=g) as sess:
    sess.run(init_op)
    
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': X_train}, y_train, num_epochs=None, shuffle=True
    )

    predict_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': X_train}, y_train, shuffle=False
    )

    predict_test_input_fn = tf.estimator.inputs.numpy_input_fn(
        {'x': X_test}, y_test, shuffle=False
    )

    embeded_text_feature_column = tf.feature_column.numeric_column(key='x', shape=512)

    checkpoint_config = tf.estimator.RunConfig(
        save_checkpoints_steps = 50,
        keep_checkpoint_max = 1,
    )

    model_dir = 'tf_models/{}'.format(target)

    estimator = tf.estimator.DNNClassifier(
        hidden_units=[128, 32],
        feature_columns=[embedded_text_feature_column],
        n_classes=5,
        dropout=0.2,
        optimizer=tf.train.ProximalAdagradOptimizer(
            learning_rate=0.01,
            l1_regularization_strengh=0.001
        ),
        model_dir=model_dir,
        config=checkpoint_config,
    )

    estimator.train(input_f=train_input_fn, steps=650)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print(test_eval_result)