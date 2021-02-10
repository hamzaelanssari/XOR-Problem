from tkinter.ttk import *
from tkinter import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from model import MultiLayerNN

''' Interface Class '''


class index:
    def __init__(self, root):
        self.root = root
        self.font = {'family': 'serif',
                     'color': '#03597C',
                     'weight': 'normal',
                     'size': 12,
                     }
        self.root.configure(bg='lightgray')
        self.root.geometry('1300x750')
        self.root.resizable(False, False)
        self.header_inputs()
        self.infos()

    def labels(self, root):
        activation_functions = ['sigmoid', 'softmax', 'relu', 'tanh', 'leaky_relu', 'sign', 'step']
        data = Label(root, text="Generate random dataset  by entering number of rows and columns",
                     font=("Goudy old style", 10, 'bold'),
                     bg="#2186A4",
                     fg="white").place(x=30, y=40)
        X = Label(root, text="Number of rows  : ",
                  font=("Goudy old style", 10, 'bold'),
                  bg="#2186A4",
                  fg="white").place(x=5, y=80)
        self.Entry_rows = Entry(root)
        self.Entry_rows.place(x=280, y=80, width=210)
        self.Entry_rows.insert(END, "300")
        y = Label(root, text="Number of columns  : ",
                  font=("Goudy old style", 10, 'bold'),
                  bg="#2186A4",
                  fg="white").place(x=5, y=120)
        self.Entry_cols = Entry(root)
        self.Entry_cols.place(x=280, y=120, width=210)
        self.Entry_cols.insert(END, "2")
        self.Entry_cols.config(state='disabled')

        epochs = Label(root, text="Number of epochs  : ",
                       font=("Goudy old style", 10, 'bold'),
                       bg="#2186A4",
                       fg="white").place(x=5, y=160)
        self.Entry_epochs = Entry(root)
        self.Entry_epochs.place(x=280, y=160, width=210)
        self.Entry_epochs.insert(END, "10000")

        lr = Label(root, text="Learning rate  : ",
                   font=("Goudy old style", 10, 'bold'),
                   bg="#2186A4",
                   fg="white").place(x=5, y=200)
        self.Entry_lr = Entry(root)
        self.Entry_lr.place(x=280, y=200, width=210)
        self.Entry_lr.insert(END, "0.001")

        input_layers = Label(root, text="Number of input layers neurons  : ",
                             font=("Goudy old style", 10, 'bold'),
                             bg="#2186A4",
                             fg="white").place(x=5, y=240)
        self.Entry_input_layers = Entry(root)
        self.Entry_input_layers.place(x=280, y=240, width=210)
        self.Entry_input_layers.insert(END, "2")
        self.Entry_input_layers.config(state='disabled')

        hidden_layers = Label(root, text="Number of hidden layers neurons  : ",
                              font=("Goudy old style", 10, 'bold'),
                              bg="#2186A4",
                              fg="white").place(x=5, y=280)
        self.Entry_hidden_layers = Entry(root)
        self.Entry_hidden_layers.place(x=280, y=280, width=210)
        self.Entry_hidden_layers.insert(END, "4")

        output_layers = Label(root, text="Number of output layers neurons  : ",
                              font=("Goudy old style", 10, 'bold'),
                              bg="#2186A4",
                              fg="white").place(x=5, y=320)
        self.Entry_output_layers = Entry(root)
        self.Entry_output_layers.place(x=280, y=320, width=210)
        self.Entry_output_layers.insert(END, "1")
        self.Entry_output_layers.config(state='disabled')

        hidden_function = Label(root, text="Hidden activation function :", font=("Goudy old style", 10, 'bold'),
                                bg="#2186A4",
                                fg="white").place(x=5, y=360)

        self.Entry_hidden_function_option = Combobox(root)
        self.Entry_hidden_function_option['values'] = activation_functions
        self.Entry_hidden_function_option.place(x=280, y=360, width=210)
        self.Entry_hidden_function_option.insert(END, "relu")

        output_function = Label(root, text="Output activation function :", font=("Goudy old style", 10, 'bold'),
                                bg="#2186A4",
                                fg="white").place(x=5, y=400)
        self.Entry_output_function_option = Combobox(root)
        self.Entry_output_function_option['values'] = activation_functions
        self.Entry_output_function_option.place(x=280, y=400, width=210)
        # self.Entry_output_function_option.delete(0, "end")
        self.Entry_output_function_option.insert(END, "sigmoid")

    def get_data(self):
        rng = np.random.RandomState(0)
        rows = int(self.Entry_rows.get())
        cols = int(self.Entry_cols.get())
        X = rng.randn(rows, cols)
        y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)
        epochs = int(self.Entry_epochs.get())
        lr = float(self.Entry_lr.get())
        inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = int(self.Entry_input_layers.get()), int(
            self.Entry_hidden_layers.get()), int(self.Entry_output_layers.get())
        hidden_activation_function = self.Entry_hidden_function_option.get()
        output_activation_function = self.Entry_output_function_option.get()
        self.draw_model(epochs, lr, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons,
                        hidden_activation_function, output_activation_function, X, y)

    def draw_model(self, epochs, lr, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons,
                   hidden_activation_function, output_activation_function, X, y):
        self.classification = Frame(self.root, bg='#BBBBBB')
        self.classification.place(x=550, y=10, height=355, width=740)
        matplotlib.use('TkAgg')
        model = MultiLayerNN(epochs, lr, inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons)
        model.activation_function(hidden_activation_function, output_activation_function)
        model.fit(X, y)
        self.draw_loss_function(model)
        self.draw_info(model)
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('#BBBBBB')
        fig.patch.set_alpha(0.6)

        ax = fig.add_subplot(111)
        ax = plot_decision_regions(X=X, y=y, clf=model, legend=2)
        canvas = FigureCanvasTkAgg(fig, master=self.classification)
        canvas.get_tk_widget().pack(expand=1)
        plt.title("XOR MLNN from scratch", fontdict=self.font)

    def draw_loss_function(self, model):
        self.loss = Frame(self.root, bg='#BBBBBB')
        self.loss.place(x=550, y=375, height=355, width=740)
        fig = plt.figure(figsize=(8, 8))
        fig.patch.set_facecolor('#BBBBBB')
        fig.patch.set_alpha(0.6)
        ax = fig.add_subplot(111)
        ax.patch.set_facecolor('#BBBBBB')
        ax.patch.set_alpha(1.0)
        ax = plt.plot(model.draw_loss())
        canvas = FigureCanvasTkAgg(fig, master=self.loss)
        canvas.get_tk_widget().pack(expand=1)
        plt.title("Loss vs Epochs", fontdict=self.font)
        # plt.xlabel("Epochs", fontdict=self.font)
        plt.ylabel("Loss", fontdict=self.font)

    def draw_info(self, model):
        accuracy_score, f1_score, recall_score, precision_score = model.info_of_classification()
        accuracy_score = "{:.2%}".format(accuracy_score)
        f1_score = "{:.2%}".format(f1_score)
        recall_score = "{:.2%}".format(recall_score)
        precision_score = "{:.2%}".format(precision_score)
        accuracy = Label(self.infos, text=str(accuracy_score),
                         font=("Goudy old style", 10, 'bold'),
                         bg="#2186A4",
                         fg="white").place(x=200, y=50)
        f1 = Label(self.infos, text=str(f1_score),
                   font=("Goudy old style", 10, 'bold'),
                   bg="#2186A4",
                   fg="white").place(x=200, y=90)
        recall = Label(self.infos, text=str(recall_score),
                       font=("Goudy old style", 10, 'bold'),
                       bg="#2186A4",
                       fg="white").place(x=200, y=130)
        precision = Label(self.infos, text=str(precision_score),
                          font=("Goudy old style", 10, 'bold'),
                          bg="#2186A4",
                          fg="white").place(x=200, y=170)

    def header_inputs(self):
        header = Frame(self.root, bg='#2186A4')
        header.place(x=10, y=10, height=485, width=500)

        header_inputs = Frame(header, bg='#03597C')
        header_inputs.place(x=0, y=0, height=30, width=500)
        txt = f'Model Configuration'
        conf = Label(header_inputs, text=txt, font=("Anaheim", 10),
                     bg="#03597C",
                     fg="white")
        conf.place(x=10, y=5)
        conf_btn = Button(header, text="Classify", command=self.get_data,
                          fg="white", bd=0,
                          bg="#03597C",
                          font=("times new roman", 11)).place(x=385, y=440, width=100, height=30)
        self.labels(header)

    def infos(self):
        self.infos = Frame(self.root, bg='#2186A4')
        self.infos.place(x=10, y=515, height=215, width=300)

        infos_inputs = Frame(self.infos, bg='#03597C')
        infos_inputs.place(x=0, y=0, height=30, width=300)
        txt = f'Model Informations'
        conf = Label(infos_inputs, text=txt, font=("Anaheim", 10),
                     bg="#03597C",
                     fg="white")
        conf.place(x=10, y=5)
        Accuracy = Label(self.infos, text="Accuracy Score  ",
                         font=("Goudy old style", 10, 'bold'),
                         bg="#2186A4",
                         fg="white").place(x=10, y=50)
        accuracy_score = Label(self.infos, text="0.0 %",
                               font=("Goudy old style", 10, 'bold'),
                               bg="#2186A4",
                               fg="white").place(x=200, y=50)
        f1 = Label(self.infos, text="F1 Score  ",
                   font=("Goudy old style", 10, 'bold'),
                   bg="#2186A4",
                   fg="white").place(x=10, y=90)
        f1_score = Label(self.infos, text="0.0 %",
                         font=("Goudy old style", 10, 'bold'),
                         bg="#2186A4",
                         fg="white").place(x=200, y=90)
        recall = Label(self.infos, text="Recall Score  ",
                       font=("Goudy old style", 10, 'bold'),
                       bg="#2186A4",
                       fg="white").place(x=10, y=130)
        recall_score = Label(self.infos, text="0.0 %",
                             font=("Goudy old style", 10, 'bold'),
                             bg="#2186A4",
                             fg="white").place(x=200, y=130)
        precision = Label(self.infos, text="Precision Score  ",
                          font=("Goudy old style", 10, 'bold'),
                          bg="#2186A4",
                          fg="white").place(x=10, y=170)
        precision_score = Label(self.infos, text="0.0 %",
                                font=("Goudy old style", 10, 'bold'),
                                bg="#2186A4",
                                fg="white").place(x=200, y=170)


''' End Interface Class '''
