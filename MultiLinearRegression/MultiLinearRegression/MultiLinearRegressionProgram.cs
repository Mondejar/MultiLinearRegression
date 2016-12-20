using System;
using MathNet.Numerics.LinearAlgebra;

/* Linear Regression code
 * 
 * Linear regression is an approach for modeling the relationship between a scalar dependent
 * variable y and one or more explanatory variables. The variable to predict is usually 
 * called the dependent variable. The predictor variables are usually called the independent variables.
 * 
 * Based on: https://msdn.microsoft.com/en-us/magazine/mt238410.aspx
 *           http://numerics.mathdotnet.com/Regression.html
 * 
 * Requires the installation of MathNet from NuGets packages
 * 
 * Author: Victor Mondejar
 * Last Date: 20/12/2016
*/

namespace MultiLinearRegression
{
    class MultiLinearRegressionProgram
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Reading data");

            Matrix<double> X, Y;
            string filename = "dataset.txt";
            //X explanatory variables 
            //Y depending variable
            read_data(filename, out X, out Y);

            // Solve for LR coefficients
            double[] LR_coeff;
            Console.WriteLine("Solving linear regression");
            LR_coeff = solve_LR(X, Y);
            Console.WriteLine("Coeff " + LR_coeff[0] + " " + LR_coeff[1] + " " + LR_coeff[2] + " " + LR_coeff[3]);

            // Calculate R-squared value
            //compute_r_squared(X, Y, LR_coeff);

            // Do a prediction
            double[] newInstance = new double[3];
            newInstance[0] = 14;
            newInstance[1] = 12;
            newInstance[2] = 0;

            double prediction = predict_value(newInstance, LR_coeff);
            Console.WriteLine("Prediction for " + newInstance[0] + " " + newInstance[1] + " " + newInstance[2] + " = " + prediction);

            Console.ReadLine();
        }

        /* 
         *  This function read the data (explanatory and dependin variable) 
         *  and creates the desing matrix 
         *  
         *  Adapt this function to your stored data
         *  In this case the data is stored like:
         *    n m
         *    variable_00 ... variable n0 prediction0
		 *    ...
         *    variable_0m ... variable nm predictionm
         *    
         *    n = number of explanatory variables
         *    m = number of instances
         * 
		*/
        static void read_data(string filename, out Matrix<double> X, out Matrix<double> Y)
        {
            // Read each line of the file into a string array. Each element
            // of the array is one line of the file.
            string[] lines = System.IO.File.ReadAllLines(filename);
            string[] first_line = lines[0].Split(null);
            int n, m;
            n = Int32.Parse(first_line[0]);
            m = Int32.Parse(first_line[1]);

            X = Matrix<double>.Build.Dense(m, n + 1);
            Y = Matrix<double>.Build.Dense(m, 1);

            string line;
            string[] elems;

            //Store the data in X, Y
            //add a column of 1.0 to X
            // 1.0 X_00 X_01 ... X_0n Y_0
            // ....
            // 1.0 X_m0 X_m1 ... X_mn Y_m
            for (int l = 1; l < lines.Length; l++)
            {
                line = lines[l];
                elems = line.Split(null);

                X[l - 1, 0] = 1.0;

                for (int nn = 0; nn < n; nn++)
                    X[l - 1, nn + 1] = Double.Parse(elems[nn]);

                Y[l - 1, 0] = Double.Parse(elems[n]);
            }
        }

        /* Main function to solve the linear regression*/
        static double[] solve_LR(Matrix<double> X, Matrix<double> Y)
        {
            int r = X.RowCount;
            int c = X.ColumnCount;

            Matrix<double> Xt, XtX, inv, invXt, mResult;

            Xt = X.Transpose();
            XtX = Xt.Multiply(X);
            inv = XtX.Inverse();
            invXt = inv.Multiply(Xt);
            mResult = invXt.Multiply(Y);

            double[] result = new double[mResult.RowCount * mResult.ColumnCount];
            for (int i = 0; i < mResult.RowCount; i++)
                for (int j = 0; j < mResult.ColumnCount; j++)
                    result[(i * mResult.ColumnCount) + j] = mResult[i, j];

            return result;
        }

        // predict the value with the computed coefficients for a new instance
        static double predict_value(double[] x, double[] coeff)
        {
            double result = coeff[0];

            for (int i = 0; i < x.Length; i++)
                result += x[i] * coeff[i + 1];

            return result;
        }

    }
}