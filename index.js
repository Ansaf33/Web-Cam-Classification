
let mobileNet;
let model;

var rockSamples = 0;
var paperSamples = 0;
var scissorSamples = 0;
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new RPSDataset();

let isPredicting = false;


async function loadMobileNet(){
  
  const mobileNet = await tf.loadLayersModel("https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json");
  const layer = mobileNet.getLayer('conv_pw_13_relu');
  return tf.model({ inputs : mobileNet.inputs , outputs : layer.output});
}


// TRAINING THE MODEL


async function train(){

  dataset.ys = null;
  dataset.encodeLabels(3);

  model = tf.sequential({
    layers:[
      tf.layers.flatten({inputShape:mobileNet.outputs[0].shape.slice(1)}),
      tf.layers.dense({units:100,activation:'relu'}),
      tf.layers.dense({units:3,activation:'softmax'})
    ]
  });

  model.compile({
    optimizer:tf.train.adam(),
    loss:'categoricalCrossentropy',
    metrics:['accuracy']
  });

  model.fit(dataset.xs,dataset.ys,{
    epochs:10,
    callbacks:{
      onBatchEnd: async(batch,logs)=>{
        loss = logs.loss.toFixed(5);
        console.log("Loss = " + loss);
      }

    }
  });


}

function handleButton(element){
  switch(element.id){
    case "0":
      rockSamples++;
      document.getElementById("rockwrite").innerText = "Rock Samples : " + rockSamples;
      break;
    case "1":
      paperSamples++;
      document.getElementById("paperwrite").innerText = "Paper Samples : " + paperSamples;
      break;

    case "2":
      scissorSamples++;
      document.getElementById("scissorwrite").innerText = "Scissor Samples : " + scissorSamples;
      break;


  }
  label = parseInt(element.id);
  const img = webcam.capture();
  dataset.addExample(mobileNet.predict(img),label);

}


async function predict(){
  while( isPredicting ){
    // FUNCTION TO FIND THE PREDICTED LABEL
    const predictedClass = tf.tidy( ()=>{
      const img = webcam.capture();
      const activation = mobileNet.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });

    const classID = (await predictedClass.data())[0];

    var text = "";

    switch(classID){
      case 0:
        text = "I see rock."
        break;

      case 1:
        text = "I see paper."
        break;

      case 2:
        text = "I see scissor."
        break;

    }
    document.getElementById("prediction").innerText = text;

    predictedClass.dispose();

    await tf.nextFrame();


  }
}

function startPrediction(){
  isPredicting = true;
  predict();
}

function stopPrediction(){
  isPredicting = false;
  predict();
}

function startTraining(){
  train();
}

async function init(){
  await webcam.setup();
  mobileNet = await loadMobileNet();
  tf.tidy(()=>(
    mobileNet.predict(webcam.capture())
  ))
  
}

init();
