var brain;
let training_data=[
   {
      inputs: [0,0],
      targets:[0]
   },
   {
      inputs: [0,1],
      targets:[1]
   },
   {
      inputs: [1,0],
      targets:[1]
   },
   {
      inputs: [1,1],
      targets:[0]
   }
];
let nn;
function setup(){
   nn=new neuralNetwork(2,16,1);
   createCanvas(400,400);
   /*var inputs=[1,0];
   var targets=[1];
   nn.train(inputs,targets);*/
   /*for(var i=0;i<1000;i++)
   {
      for(data of training_data)
      {
         nn.train(data.inputs,data.targets)
      }
   }
  console.table(nn.feedforward([0,0]));
  console.table(nn.feedforward([1,0]));
  console.table(nn.feedforward([0,1]));
  console.table(nn.feedforward([1,1]));*/
   //  console.log(output);
   //noStroke()
}
function draw(){
   background(0); 
   for(var i=0;i<10000;i++)
   {
     for(data of training_data)
      {
         nn.train(data.inputs,data.targets)
      }
   }
   let resolution=10;
   let cols=width/resolution;
   let rows=height/resolution;
   for(let i=0;i<cols;i++)
   {
      for(let j=0;j<rows;j++)
      {
         let x1=i/cols;
         let x2=j/rows;
         let inputs=[x1,x2];
         let y=nn.feedforward(inputs); 
         fill(y*255);
         //fill(random(255));
         rect(i*resolution,j*resolution,resolution,resolution)
      }
   }

}