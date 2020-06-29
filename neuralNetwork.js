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
        
        let output=Matrix.multiply(this.weights_ho,hidden);
        output.add(this.bias_o);
        output.map(sigmoid);

        return output.toArray(); 
    }
    train(inputs,targets){
        inputs=Matrix.fromArray(inputs);
        let hidden=Matrix.multiply(this.weights_ih,inputs);
        hidden.add(this.bias_h);
        hidden.map(sigmoid);

        let outputs=Matrix.multiply(this.weights_ho,hidden);
        outputs.add(this.bias_o);
        outputs.map(sigmoid); 
        targets=Matrix.fromArray(targets);
        let output_errors=Matrix.subtract(targets,outputs);
        outputs.map(dsigmoid);
        outputs=Matrix.multiply(outputs,output_errors);
        outputs=Matrix.multiply(this.lr,outputs);
        this.bias_o.add(outputs);
        let hidden_t=Matrix.transpose(hidden);
        let change_in_weights_ho=Matrix.multiply(outputs,hidden_t);
        this.weights_ho.add(change_in_weights_ho);
        let weights_ho_t=Matrix.transpose(this.weights_ho);
       
         let hidden_errors=Matrix.multiply(weights_ho_t,output_errors);
        let hidden_gradient=Matrix.map(hidden, dsigmoid);
        let b=Matrix.multiply(hidden_gradient,hidden_errors);   
        b=Matrix.multiply(this.lr,b);
        this.bias_h.add(b);
        let inputs_t=Matrix.transpose(inputs);
        
        let change_in_weights_ih=Matrix.multiply(b,inputs_t);
        this.weights_ih.add(change_in_weights_ih);

    } 
} 