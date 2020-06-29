class Matrix{
    constructor(rows,cols){
    this.rows=rows;
    this.cols=cols;
    this.matrix=[];
    for(let i=0;i<this.rows;i++)
    {
        this.matrix[i]=[];
        for(let j=0;j<this.cols;j++)
        {
            this.matrix[i][j]=0;
        }
    }
}
static fromArray(arr){
    let m=new Matrix(arr.length,1)
    for(let i=0;i<arr.length;i++)
    {
        m.matrix[i][0]=arr[i];
    }
    return m;
}
toArray(){
    let arr=[];
    for(let i=0;i<this.rows;i++)
        {
            for(let j=0;j<this.cols;j++)
            {
                arr.push(this.matrix[i][j]);
            }
        }
        return arr;

}
randomize()
{
    for(let i=0;i<this.rows;i++)
        {
            for(let j=0;j<this.cols;j++)
            {
                this.matrix[i][j]=Math.random()*2-1;
            }
        }
}
add(n)
{
    if(n instanceof Matrix){
       /* console.log('jiorsjthjrtshkljklh');
        console.log(this.rows);
        console.log(this.cols);
        console.log(n.rows);
        console.log(n.cols);*/
        
        for(let i=0;i<this.rows;i++)
        {
            for(let j=0;j<this.cols;j++)
            {
                this.matrix[i][j]+=n.matrix[i][j];
            }
        }
    }
    else{
        for(let i=0;i<this.rows;i++)
        {
            for(let j=0;j<this.cols;j++)
            {
                this.matrix[i][j]+=n;
            }
        }
    }
}
static subtract(a,b){
    
    let result=new Matrix(b.rows,b.cols);
    for(let i=0;i<result.rows;i++)
        {
            for(let j=0;j<result.cols;j++)
            {
                result.matrix[i][j]=a.matrix[i][j]-b.matrix[i][j];
            }
        }
        return result;
    
}
static transpose(m){
    let result=new Matrix(m.cols,m.rows);
    for(let i=0;i<m.rows;i++)
    {
        for(let j=0;j<m.cols;j++)
        {
            result.matrix[j][i]=m.matrix[i][j];
        }
    }
    return result;
}
static multiply(m1,m2){
    if(m2 instanceof Matrix && m1 instanceof Matrix)
    {
    
        {
            let result=new Matrix(m1.rows,m2.cols);

            for(let i=0;i<m1.rows;i++)
            {
                for(let j=0;j<m2.cols;j++)
                {
                    for(let k=0;k<m1.cols;k++)
                    {
                        result.matrix[i][j]+=m1.matrix[i][k]*m2.matrix[k][j]
                    }
                }
            } 
            return result;
        } 
    }
    else{
        let result=new Matrix(m2.rows,m2.cols);
        for(let i=0;i<m2.rows;i++)
            {
                for(let j=0;j<m2.cols;j++)
                {
                    result.matrix[i][j]=m1*m2.matrix[i][j];
                }
            }
            return result; 

    }
}
map(func){ 
   
    for(let i=0;i<this.rows;i++)
        {
            for(let j=0;j<this.cols;j++)
            {
                let value=this.matrix[i][j];
                this.matrix[i][j] =func(value);
            } 
        }
}
static map(m,func)
{
    let result=new Matrix(m.rows,m.cols);
    for(let i=0;i<m.rows;i++)
    {
        for(let j=0;j<m.cols;j++)
        {
            
            let value=m.matrix[i][j];
            result.matrix[i][j] =func(value);
        } 
    }
    return result;
}

print(){
    console.table(this.matrix);
}
  

}