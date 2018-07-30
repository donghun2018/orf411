/******************************************************************************
  *  Name:    Joy Hii
  *  NetID:   jhii
  * 
  *  Description:  
  ******************************************************************************/
import edu.princeton.cs.algs4.StdRandom;
import edu.princeton.cs.algs4.RedBlackBST;

public class AssetSellingModel {
    
    /*
    // array to hold the state variables, with array indices corresponding to time, t
    private final StateVar[] state_vars; 
    private final DecisionVar[] dec_vars;  
    private final ExInfo[] ex_info;
    */
    
    // t denotes time
    private int t;
    // binary search tree to store state variables
    private final RedBlackBST<Integer, StateVar> state_vars; 
    // binary search tree to store decision variables
    private final RedBlackBST<Integer, DecisionVar> dec_vars; 
    // binary search tree to store exogenous information 
    private final RedBlackBST<Integer, ExInfo> ex_info;  

    // state variable consists of physical state and price per share
    private class StateVar {
        private int R;  // whether we're holding the stock or not
        private double p;  // the price per share at time t
        
        // StateVar constructor
        public StateVar(int physical_state, double share_price) {
            this.R = physical_state;
            this.p = share_price;
        }
    }
    
    // decision variable which is defined by chosen decision policy
    private class DecisionVar {
        private int x; // whether we sell the stock or not
        private StateVar s; // consists of physical state and price per share
        
        // DecisionVar constructor - not sure if I need statevar here
        public DecisionVar(int decision, StateVar s) {
            if (decision > s.p) throw new IllegalArgumentException();
            this.x = decision;
            this.s = s;
        }
    }
    
    // implement some chosen policy X^pi(S_t)
    // private method to implement sell-low policy 
    private DecisionVar sellLow(DecisionVar x0, double limit) {
        DecisionVar x1 = x0;
        if (x0.s.p < limit && x0.s.R == 1) x1.x = 1;
        // don't know how to code "if t=T and R_t = 1" // else if 
        else x1.x = 0;
        return x1;
    }
    
    // private method to implement high-low policy 
    
    
    // private method to implement track policy 
    
    // exogenous information is change in price
    private class ExInfo {
        private double w; // change in price or new price of stock
        
        // ExInfo constructor
        public ExInfo(double w) {
            this.w = w;
        }
    }
    
    // private method to calculate Gaussian pdfs
    // return pdf(x) = standard Gaussian pdf
    private double pdf(double x) {
        return Math.exp(-x*x / 2) / Math.sqrt(2 * Math.PI);
    }
    
    // return pdf(x, mu, signma) = Gaussian pdf with mean mu and stddev sigma
    private double pdf(double x, double mu, double sigma) {
        return pdf((x - mu) / sigma) / sigma;
    }
    
    // 1st way that lets W_t be change in price
    // (assumes a probabilistic model for price change)
    private ExInfo priceChange() {
        double U = StdRandom.uniform();
        double W = pdf(U, 0, 1);
        ExInfo w1 = new ExInfo(W);
        return w1;
    }
    
    // 2nd way that lets W_t be the new price
    private ExInfo newPrice() {
        ExInfo w1 = new ExInfo(state_vars.get(t).p);
        return w1;
    }
    
    public AssetSellingModel() { 
        t = 0;
        state_vars = new RedBlackBST<Integer, StateVar>(); 
        dec_vars = new RedBlackBST<Integer, DecisionVar>(); 
        ex_info = new RedBlackBST<Integer, ExInfo>(); 
        
        /*
        // the process continues until we sell the stock
        while (state_vars.get(t).R == 1) {
            
        }
        */
        
    }
    /*
    // private method to resize array 
    private void resize(int capacity) { 
        assert capacity >= N; 
        Item[] temp = (Item[]) new Object[capacity]; 
        for (int i = 0; i < N; i++) { 
            temp[i] = a[i]; 
        } 
        a = temp; 
    } 
    */
    
    // transition function takes in state variable, decision variable and exogenous 
    // info to describe how the state evolves
    public StateVar transitionFunction(StateVar s0, 
                                       DecisionVar x0, ExInfo w0) {
        StateVar s1 = new StateVar(s0.R - x0.x, s0.p + w0.w);
        return s1;
    }
    
    /*
    // objective function
    public StateVar objectiveFunction(StateVar s0, 
                                       DecisionVar x0, ExInfo w0) {

      
    }
    
    // contribution function
    public double contributionFunction(StateVar s0, DecisionVar x0) {
        double contribution = s0.p * x0.x;
        return contribution;
    }
    */
    
   
    // unit testing
    public static void main(String[] args) {
    }  
    
}