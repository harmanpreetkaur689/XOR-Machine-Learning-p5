function sigmoid(x)
{
    return 1/(1+Math.exp(-x));
}
function dsigmoid(x){
    return x*(1-x);
}
class neuralNetwork
{
    constructor(input_n,hidden_n,outputs_n)
    {
        this.input_n=input_n;
        this.hidden_n=hidden_n;
        this.outputs_n=outputs_n;

       // this.weights_ih = this.input_nodes.weights_ih.copy();
      //this.weights_ho = this.input_nodes.weights_ho.copy();

        this.weights_ih=new Matrix(this.hidden_n,this.input_n) 
        this.weights_ho=new Matrix(this.outputs_n,this.hidden_n)
       
        this.weights_ih.randomize();
        this.weights_ho.randomize(); 
        
        this.bias_h=new Matrix(this.hidden_n,1);
        this.bias_o=new Matrix(this.outputs_n,1);
        this.bias_h.randomize();
        this.bias_o.randomize();
        this.lr=0.1;
    }
    copy() {
        return new NeuralNetwork(this);
      }
    
    feedforward(inputs){
        inputs=Matrix.fromArray(inputs);
        
        let hidden=Matrix.multiply(this.weights_ih,inputs);
        hidden.add(this.bias_h);
        hidden.map(sigmoid);
        //hidden.map(dsigmoid);


        let output=Matrix.multiply(this.weights_ho,hidden);
        output.add(this.bias_o);
        //console.log(output)
        output.map(sigmoid);
        //output.map(dsigmoid);


        return output.toArray(); 
    }
    train(inputs,targets){
        inputs=Matrix.fromArray(inputs);
        //console.table(inputs.matrix);
        //console.table(this.weights_ih.matrix)
        let hidden=Matrix.multiply(this.weights_ih,inputs);
        //console.table(hidden.matrix)
        //console.table(this.bias_o.matrix)
        //console.table(this.bias_h.matrix)
        hidden.add(this.bias_h);
        //console.table(hidden.matrix);
        hidden.map(sigmoid);
        // console.table(this.weights_ho.matrix);

        let outputs=Matrix.multiply(this.weights_ho,hidden);
          //console.table(outputs.matrix);
          //console.table(this.bias_o.matrix);
        outputs.add(this.bias_o);
       // console.table(outputs.matrix);
        //console.table(this.bias_o.matrix);
        outputs.map(sigmoid); 
       // console.table(outputs.matrix);
        targets=Matrix.fromArray(targets);
        let output_errors=Matrix.subtract(targets,outputs);
        //console.table(output_errors.matrix)
        outputs.map(dsigmoid);
        //console.table(outputs.matrix)
        outputs=Matrix.multiply(outputs,output_errors);
        //console.table(outputs.matrix)
        outputs=Matrix.multiply(this.lr,outputs);
        this.bias_o.add(outputs);
       // console.table(outputs.matrix);
        let hidden_t=Matrix.transpose(hidden);
        //console.table(hidden_t.matrix)
        let change_in_weights_ho=Matrix.multiply(outputs,hidden_t);
      //  console.table(change_in_weights_ho)
       // console.table(this.weights_ho.matrix)
        this.weights_ho.add(change_in_weights_ho);
        let weights_ho_t=Matrix.transpose(this.weights_ho);
        //console.table(this.weights_ho.matrix)
        


        let hidden_errors=Matrix.multiply(weights_ho_t,output_errors);
        let hidden_gradient=Matrix.map(hidden, dsigmoid);
        //console.table(hidden_errors.matrix)
        //console.table(hidden_gradient.matrix)
        let b=Matrix.multiply(hidden_gradient,hidden_errors);
        //console.table(b.matrix)
        b=Matrix.multiply(this.lr,b);
        //console.table(b.matrix)
        this.bias_h.add(b);
        let inputs_t=Matrix.transpose(inputs);
        //console.table(inputs_t.matrix);
        //console.table(b.matrix);
        let change_in_weights_ih=Matrix.multiply(b,inputs_t);
       // console.table(change_in_weights_ih.matrix)
        //console.table(this.weights_ih.matrix)
        //console.table(change_in_weights_ih.matrix)
        this.weights_ih.add(change_in_weights_ih);
       // this.weights_ih=Matrix.transpose(this.weights_ih);
        //console.table(this.weights_ih.matrix)

        //this.weights_ho=Matrix.transpose(this.weights_ho);
      //  this.weights_ih=Matrix.transpose(this.weights_ih);
        
        //outputs.print();
        //targets.print();
        //output_errors.print();

    } 
} 