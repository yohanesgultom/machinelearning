package org.yohanes.machinelearning.boltzmann.xor;

public class Main {

	public static void main(String[] args) {
		
		System.out.println("Boltzmann Machine XOR");
		System.out.println("=====================");
		
		// training data
		boolean[][] trainingData = {
				{false, false, false ^ false},
				{true, false, true ^ false},
				{true, true, true ^ true},
				{false, true, false ^ true}
		};		
		
		double temp = 100;
		double tempDecrement = 0.9;
		int m = 20;
		double learningRate = 0.75;
		double threshold = 0.065;
		try {
			if (args.length == 5) {
				temp = Double.parseDouble(args[0]);
				tempDecrement = Double.parseDouble(args[1]);
				m = Integer.parseInt(args[2]);
				learningRate = Double.parseDouble(args[3]);
				threshold = Double.parseDouble(args[4]);
			} 
		} catch (Exception e) {
			
		} 
		// init instance
		BoltzmannMachineXOR bm = new BoltzmannMachineXOR(temp, // initial temperature 
				tempDecrement, // temperature decrement
				m, // number of times given for network relaxation during annealing
				learningRate, // learning rate
				trainingData, // training data
				threshold // threshold
			);
		
		bm.train();
		bm.printWeights();
		
		// test
		int n = 5;
		String testFormat = "expected = %b ^ %b = %b | result = %b";
		for (int i=0;i<n;i++) {
			System.out.println("Test #" + i);
			System.out.println(String.format(testFormat, true, true, true ^ true, bm.xor(true, true)));
			System.out.println(String.format(testFormat, true, false, true ^ false, bm.xor(true, false)));
			System.out.println(String.format(testFormat, false, true, false ^ true, bm.xor(false, true)));
			System.out.println(String.format(testFormat, false, false, false ^ false, bm.xor(false, false)));
			System.out.println();
		}
		System.out.println("Bye");
	}

}

class BoltzmannMachineXOR {
	private int units;
	
	private int[] inputUnits;
	private int[] outputUnits;
	private int[] hiddenUnits;
	
	private boolean output[];
	private double weight[][];
	private double temperature;
	private double temperatureStep;
	private double learningRate;
	private boolean[][] trainingData;
	private int m;
	private double threshold;
	
	public BoltzmannMachineXOR(double temperature, double temperatureStep, int m, double learningRate, boolean[][] data, double threshold) {
		this.temperature = temperature;
		this.temperatureStep = temperatureStep;
		this.learningRate = learningRate;
		this.threshold = threshold;
		
		// TODO parameterize
		// hard-coded units arrangement for XOR case
		// XOR requires a I/O network architectures where 
		// input units/nodes's weights are unidirectional AND there is no connections between input units
		this.units = 4; 
		this.inputUnits = new int[]{0, 1};
		this.outputUnits = new int[]{2};
		this.hiddenUnits = new int[]{3};
		
		this.output = new boolean[this.units];
		this.weight = new double[this.units][this.units];
		this.trainingData = data;
		this.m = m;
		
		// init weight = 1
		int defaultWeight = 1;
		for (int i=0;i<this.units;i++) {
			for (int j=0;j<this.units;j++) {
				if (i == j) {
					// identical
					this.weight[i][j] = 0;					
				} else if (belongsTo(j, this.inputUnits)) {
					// no connection between input units in I/O architecture
					this.weight[i][j] = 0;
				} else {
					this.weight[i][j] = defaultWeight;
				}
			}
		}
		
		//printMatrix(this.weight, "w");
					
	}	

	/**
	 * Train boltzmann machine
	 * @return
	 */
	public void train() {
		
		// average of delta of weights
		double dW = 999; // dummy init
		int cycle = 0;
		do {
			
			double[][] positiveProbability = new double[this.units][this.units];
			double[][] negativeProbability = new double[this.units][this.units];				

			// iterate training data
			for (int n = 0; n < this.trainingData.length; n++) {						
				// init Visible units with training data
				// let the Hidden unit initialized with false = 0 
				for (int i=0; i<this.trainingData[n].length; i++) {				
					this.output[i] = this.trainingData[n][i];
				}
				// anneal and collect p+ for each connected non-clamped units (hidden units) 
				positiveProbability = addMatrix(positiveProbability, this.annealHiddenUnitOnly(this.m));
			}					
			
			// calculate average p+
			positiveProbability = divideMatrix(positiveProbability, this.trainingData.length);
						
			// randomize inputs
			this.output = new boolean[this.units];
			this.randomizeInputUnits();
			
			// anneal hidden and output units
			// calculate p- for each non-clamped connected units (hidden and output units)
			negativeProbability = this.annealHiddenAndOutput(this.m);
						
			// update weights
			dW = this.updateWeights(positiveProbability, negativeProbability);			
			System.out.println(String.format("dW = %f", dW));
			
//			printMatrix(positiveProbability, "p+");
//			printMatrix(negativeProbability, "p-");
			
			cycle++;
			
		} while (dW > this.threshold);			

		System.out.println(String.format("Training done. dW = %f | cycle = %d", dW, cycle));

	}

	/**
	 * Get x XOR y
	 * @param x
	 * @param y
	 * @return
	 */
	public boolean xor(boolean x, boolean y) {
		this.output = new boolean[this.units];
		this.output[this.inputUnits[0]] = x;
		this.output[this.inputUnits[1]] = y;				
		this.annealHiddenAndOutputWithRandomUpdate(this.m);	
		return this.output[this.outputUnits[0]];
	}

	/**
	 * Do annealing only on hidden units
	 * @param m
	 * @return
	 */
	private double[][] annealHiddenUnitOnly(int m) {
		return this.anneal(this.hiddenUnits, m);
	}

	/**
	 * Do annealing on hidden and output units
	 * @param m
	 * @return
	 */
	private double[][] annealHiddenAndOutput(int m) {
		return this.anneal(concat(this.hiddenUnits, this.outputUnits), m);
	}

	private void annealHiddenAndOutputWithRandomUpdate(int m) {
		// reduce temperature step by step			
		for (double t = this.temperature; t > 0; t = t - this.temperatureStep) {
			int[] onCount = new int[this.units];
			for (int n=0;n<m;n++) {
				// activate random target (between output and hidden)
				int randomUnit = (Math.random() > 0.5) ? this.hiddenUnits[0] : this.outputUnits[0];
				//System.out.println("randomUnit = " + randomUnit);
				this.activate(randomUnit, t);
				if (this.output[randomUnit]) onCount[randomUnit]++;				
			}			
			for (int i=0;i<this.units;i++) {
				if (onCount[i] > 0) {
					double p = (double) onCount[i] / m;
					this.output[i] = p >= 0.5;
				}
			}
		}
	}

	/**
	 * Do annealing m times on unit in given targets
	 * @param targets
	 * @param m
	 * @return
	 */
	private double[][] anneal(int[] targets, int m) {
		double[][] onConnectedCount = new double[this.units][this.units];
		// reduce temperature step by step			
		int count = 0;
		for (double t = this.temperature; t > 0; t = t - this.temperatureStep) {
			int[] onCount = new int[this.units];
			// enough time (m) must be given to reach equilibrium
			for(int n=0;n<m;n++) {
				// activate given targets
				for (int a=0;a<targets.length;a++) {
					this.activate(targets[a], t);
				}
				// count ON state
				for (int i=0;i<this.units;i++) {
					if (this.output[i]) onCount[i]++;
					for (int j=0;j<this.units;j++) {
						if (i != j && this.output[i] && this.output[j] && !belongsTo(j, this.inputUnits)) {
							onConnectedCount[i][j]++;
						}
					}
				}
			}

			// set output with most probable value
			for (int i=0;i<this.units;i++) {
				if (onCount[i] > 0) {
					double p = (double) onCount[i] / m;
					this.output[i] = p >= 0.5;
				}
			}

			
			count++;
		}
		return divideMatrix(onConnectedCount, count * m);
	}

	/**
	 * Invoke activation function for unit j with temperature t
	 * Update output[j] based on probability
	 * @param j
	 * @param t
	 */
	private void activate(int j, double t) {		
		double s = this.sumWeights(j);
		double p = 1 / (1 + Math.exp(-s / t));	
		double rand = Math.random();
		this.output[j] = (p > rand);
	}

	/**
	 * Sum input value * weight to node j
	 * @param j
	 * @return
	 */
	private double sumWeights(int j) {
		double s = 0;
		for (int i=0;i<this.units;i++) {			
			s += this.weight[i][j] * (this.output[i] ? 1 : 0);
		}
		return s;
	}
		
	/**
	 * Update unit weights
	 * @param probabilityPositive
	 * @param probabilityNegative
	 * @return
	 */
	private double updateWeights(double[][] probabilityPositive, double[][] probabilityNegative) {
		double sumDW = 0;
		for (int i=0;i<this.units;i++) {
			for (int j=0;j<this.units;j++) {
				double dW = probabilityPositive[i][j] - probabilityNegative[i][j];
				this.weight[i][j] = this.weight[i][j] + (this.learningRate * dW);
				sumDW += Math.abs(dW);
			}
		}		
		return (double) sumDW / (this.units * this.units);
	}
	
	/**
	 * Get random input
	 */
	private void randomizeInputUnits() {
		for (int i:this.inputUnits) {
			this.output[i] = Math.random() >= 0.5;
		}
	}
	
	/**
	 * Print weights
	 */
	public void printWeights() {
		printMatrix(this.weight, "Weight");
	}
		
	/**
	 * Print matrix
	 * @param matrix
	 * @param label
	 */
	public static void printMatrix(double[][] matrix, String label) {
		for (int i=0;i<matrix.length;i++) {
			for (int j=0;j<matrix[i].length;j++) {
				System.out.println(String.format("%s[%d][%d]=%f", label, i, j, matrix[i][j]));
			}
		}
	}
		
	/**
	 * Check if a unit/node index belong to certain unit group
	 * @param index
	 * @param units
	 * @return
	 */
	public static boolean belongsTo(int index, int[] units) {
		boolean found = false;
		for (int u:units) {
			if (index == u) {
				found = true;
				break;
			}
		}
		return found;
	}

	/**
	 * Concat two arrays
	 * @param a
	 * @param b
	 * @return
	 */
	public static int[] concat(int[] a, int[] b) {
		int[] res = new int[a.length + b.length];
		int i = 0;
		for (int j=0;j<a.length;j++) {
			res[i] = a[j]; 
			i++;	
		}
		for (int j=0;j<b.length;j++) {
			res[i] = b[j]; 
			i++;	
		}
		return res;
	}
	
	/**
	 * Add 2 matrices
	 * @param p1
	 * @param p2
	 * @return
	 */
	public static double[][] addMatrix(double[][] p1, double[][] p2) {
		double[][] res = new double[p1.length][p1.length];
		for (int i=0;i<res.length;i++) {
			for (int j=0;j<res.length;j++) {
				res[i][j] = p1[i][j] + p2[i][j];
			}
		}
		return res;
	}

	/**
	 * Divide matrix m with divider
	 * @param m
	 * @param divider
	 * @return
	 */
	public static double[][] divideMatrix(double[][] m, int divider) {
		double[][] res = new double[m.length][m.length];
		for (int i=0;i<res.length;i++) {
			for (int j=0;j<res.length;j++) {
				res[i][j] = (double) m[i][j] / divider;
			}
		}
		return res;
		
	}
	
}