# Background and Basics</br>

### What are `Channels` and `Kernels`?</br>
**`Channels`** come from `media`. Looking at broadcast technology behind TVs you have mulitple channels for different information that gets broadcasted to your TV. Let's assume that we are talking about 2D convolutions applied on images.</br>

In a grayscale image, the data is a matrix of dimensions **_w×h_**, where **_w_** is the `width` of the image and **_h_** is its `height`. In a color image, we normally have 3 channels: `red, green and blue` ; this way, a color image can be represented as matrix of dimensions **_w×h×c_**, where **_c_** is the `number of channels`, that is, 3.
    
A convolution layer receives the image **_w×h×c_** as input, and generates as output an activation map of dimensions **_w′×h′×c′_**. The number of input channels in the convolution is c, while the number of output channels is **_c′_** . The filter for such a convolution is a tensor of dimensions **_f×f×c×c′_** , where **_f_** is the filter size (normally 3 or 5).

This way, the number of channels is the `depth of the matrices involved in the convolutions`. Also, a convolution operation defines the variation in such depth by specifying input and output channels.

<img src="../scenarios/media/kernels.jpg" align="right" alt="" width="300"/> </br>
These explanations are directly extrapolable to 1D signals or 3D signals, but the analogy with image channels made it more appropriate to use 2D signals in the example.

**`Kernels`** are nothing but a filter that is used to extract the features from the images. The kernel is a matrix that moves over the input data, performs the dot product with the sub-region of input data, and gets the output as the matrix of dot products. Kernel moves on the input data by the stride value. If the stride value is 2, then kernel moves by 2 columns of pixels in the input matrix. In short, the kernel is used to extract high-level features like edges from the image.

### Why should we (nearly) `always use 3x3 kernels`?</br>
This question can be answered in two parts. `One part` is to answer `why noy even kernels` _(2x2, 4x4)_ and the `second part` is why not using `bigger kernels`_(5x5, 7x7, 9x9..)_ .

First, with even kernels the problem is its difficult to find axis of symmetry. Without centre point, it is difficult to depict information in a symmetric way.

Second, using a `higher size kernel increases the computation cost` with more number of parameters and also the amount of information or features extracted are considerably lesser (as the dimension of next layer reduces greatly). Using a `lower size kernel like 1x1 does not account of features from the neighbouring pixels`, 1x1 is used only in cases of reducing the dimensions.

3x3 is the smallest unit which can be used to compute any kernel size output and seems to be a best fit. If we need 5x5 kernel output, we can convolve with _3x3 twice (3x3 + 3x3 = 18 parameter)_ and if we need 7x7 output, we can convolve using _3x3 thrice (33 + 33 + 3*3 = 27 parameters)_ and so on. And GPUs have accelerated 3x3 operation, so it is much faster to perform the convolution using 3x3 kernel.

### How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)</br>

<img src="../scenarios/media/cnn.gif" align="right" alt="" width="300"/> </br>
Each time, when a 3x3 convolution is performed, we end up with 2 pixels lesser output channel. When we perform 3x3 on 5x5 image, we get a 3x3 image.

Without Max-pooling, **99 times** 3x3 convolution needs to be performed on 199x199 to reach 1x1 image!

|  Operation-No | Image O/P	|  Operation-No | Image O/P	|  Operation-No | Image O/P	|
|---------------|-----------|---------------|-----------|---------------|-----------|
|		1		|	197X197	|		2		|	195X195	|		3		|	193X193	|
|		4		|	191X191	|		5		|	189X189	|		6		|	187X187	|
|		7		|	185X185	|		8		|	183X183	|		9		|	181X181	|
|		10		|	179X179	|		11		|	177X177	|		12		|	175X175	|
|		13		|	173X173	|		14		|	171X171	|		15		|	169X169	|
|		16		|	167X167	|		17		|	165X165	|		18		|	163X163	|
|		19		|	161X161	|		20		|	159X159	|		21		|	157X157	|
|		22		|	155X155	|		23		|	153X153	|		24		|	151X151	|
|		25		|	149X149	|		26		|	147X147	|		27		|	145X145	|
|		28		|	143X143	|		29		|	141X141	|		30		|	139X139	|
|		31		|	137X137	|		32		|	135X135	|		33		|	133X133	|
|		34		|	131X131	|		35		|	129X129	|		36		|	127X127	|
|		37		|	125X125	|		38		|	123X123	|		39		|	121X121	|
|		40		|	119X119	|		41		|	117X117	|		42		|	115X115	|
|		43		|	113X113	|		44		|	111X111	|		45		|	109X109	|
|		46		|	107X107	|		47		|	105X105	|		48		|	103X103	|
|		49		|	101X101	|		50		|	99X99	|		51		|	97X97	|
|		52		|	95X95	|		53		|	93X93	|		54		|	91X91	|
|		55		|	89X89	|		56		|	87X87	|		57		|	85X85	|
|		58		|	83X83	|		59		|	81X81	|		60		|	79X79	|
|		61		|	77X77	|		62		|	75X75	|		63		|	73X73	|
|		64		|	71X71	|		65		|	69X69	|		66		|	67X67	|
|		67		|	65X65	|		68		|	63X63	|		69		|	61X61	|
|		70		|	59X59	|		71		|	57X57	|		72		|	55X55	|
|		73		|	53X53	|		74		|	51X51	|		75		|	49X49	|
|		76		|	47X47	|		77		|	45X45	|		78		|	43X43	|
|		79		|	41X41	|		80		|	39X39	|		81		|	37X37	|
|		82		|	35X35	|		83		|	33X33	|		84		|	31X31	|
|		85		|	29X29	|		86		|	27X27	|		87		|	25X25	|
|		88		|	23X23	|		89		|	21X21	|		90		|	19X19	|
|		91		|	17X17	|		92		|	15X15	|		93		|	13X13	|
|		94		|	11X11	|		95		|	9X9	    |		96		|	7X7	    |
|		97		|	5X5	    |		98		|	3X3	    |		99		|	1X1	    |


### How are kernels initialized?</br>

### What happens during the training of a DNN?</br>
<html>
<head>
<meta charset="utf-8">
<meta http-equiv="X-UA-Compatible" content="IE=edge">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="author" content="Jingru Guo">
<link rel="canonical" href="https://www.deeplearning.ai/ai-notes/initialization/" />
<title>Initializing neural networks - deeplearning.ai</title>
<meta property="og:title" content="AI Notes: Initializing neural networks - deeplearning.ai" />
<meta property="og:type" content="article" />
<meta property="og:url" content="https://www.deeplearning.ai/ai-notes/initialization/" />
<meta property="og:image" content="https://www.deeplearning.ai/ai-notes/assets/images/layout/ai-notes-og-image.png" />
<meta property="og:site_name" content="deeplearning.ai" />
<meta property="og:description" content="AI Notes: Initializing neural networks - deeplearning.ai" />

<link rel="stylesheet" href="https://use.fontawesome.com/releases/v5.5.0/css/all.css" integrity="sha384-B4dIYHKNBt8Bc12p+WXckhzcICo0wtJAoU8YZTY5qE0Id1GSseTk6S+L3BlXeVIU" crossorigin="anonymous">
<style id="" media="all">/* hebrew */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 300;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtrhnIGSV35Gu.woff2) format('woff2');
  unicode-range: U+0590-05FF, U+20AA, U+25CC, U+FB1D-FB4F;
}
/* latin-ext */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 300;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtrhnIGiV35Gu.woff2) format('woff2');
  unicode-range: U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
}
/* latin */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 300;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtrhnIGaV3w.woff2) format('woff2');
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
/* hebrew */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 400;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtuZnIGSV35Gu.woff2) format('woff2');
  unicode-range: U+0590-05FF, U+20AA, U+25CC, U+FB1D-FB4F;
}
/* latin-ext */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 400;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtuZnIGiV35Gu.woff2) format('woff2');
  unicode-range: U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
}
/* latin */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 400;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtuZnIGaV3w.woff2) format('woff2');
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
/* hebrew */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 600;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtjhgIGSV35Gu.woff2) format('woff2');
  unicode-range: U+0590-05FF, U+20AA, U+25CC, U+FB1D-FB4F;
}
/* latin-ext */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 600;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtjhgIGiV35Gu.woff2) format('woff2');
  unicode-range: U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
}
/* latin */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 600;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtjhgIGaV3w.woff2) format('woff2');
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
/* hebrew */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 700;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtgFgIGSV35Gu.woff2) format('woff2');
  unicode-range: U+0590-05FF, U+20AA, U+25CC, U+FB1D-FB4F;
}
/* latin-ext */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 700;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtgFgIGiV35Gu.woff2) format('woff2');
  unicode-range: U+0100-024F, U+0259, U+1E00-1EFF, U+2020, U+20A0-20AB, U+20AD-20CF, U+2113, U+2C60-2C7F, U+A720-A7FF;
}
/* latin */
@font-face {
  font-family: 'Assistant';
  font-style: normal;
  font-weight: 700;
  src: url(/fonts.gstatic.com/s/assistant/v7/2sDPZGJYnIjSi6H75xkZZE1I0yCmYzzQtgFgIGaV3w.woff2) format('woff2');
  unicode-range: U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+2000-206F, U+2074, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD;
}
</style>

<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.11.0/jquery.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>

<link rel="shortcut icon" type="image/png" href="/ai-notes/assets/images/layout/favicon.png" />

<link rel="stylesheet" href="/ai-notes/assets/css/template.css">

<link rel="stylesheet" href="/ai-notes/assets/css/article.css">

<script src="https://d3js.org/d3.v5.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>

<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@0.13.3/dist/tf.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.9.0/dist/katex.min.css" integrity="sha384-TEMocfGvRuD1rIAacqrknm5BQZ7W7uWitoih+jMNFXQIbNl16bO8OZmylH/Vi/Ei" crossorigin="anonymous">
<script src="https://cdn.jsdelivr.net/npm/katex@0.9.0/dist/katex.min.js" integrity="sha384-jmxIlussZWB7qCuB+PgKG1uLjjxbVVIayPJwi6cG6Zb4YKq0JIw+OMnkkEC7kYCq" crossorigin="anonymous" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.10.2/dist/contrib/auto-render.min.js" integrity="sha384-kWPLUVMOks5AQFrykwIup5lo0m3iMkkHrD0uJ4H5cjeGihAutqP0yW0J6dpFiVkI" crossorigin="anonymous" onload="renderMathInElement(document.body, {delimiters: [
            {left: '$$', right: '$$', display: true},
            {left: '$', right: '$', display: false}
    ]});" type="da8b9b91809b182b01d95155-text/javascript">
</script>

<script src="/ai-notes/assets/js/TweenMax.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/Draggable.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/DrawSVGPlugin.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/MorphSVGPlugin.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/ThrowPropsPlugin.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/snap.svg-min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>

<script src="/ai-notes/assets/js/cppn.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/d3.tip.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/tool.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="https://d3js.org/d3-contour.v1.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="https://d3js.org/d3-scale-chromatic.v1.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3-legend/2.25.6/d3-legend.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.13.1/styles/monokai-sublime.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/9.13.1/highlight.min.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script type="da8b9b91809b182b01d95155-text/javascript">hljs.initHighlightingOnLoad();</script>

<script type="da8b9b91809b182b01d95155-text/javascript">
	!function(){var analytics=window.analytics=window.analytics||[];if(!analytics.initialize)if(analytics.invoked)window.console&&console.error&&console.error("Segment snippet included twice.");else{analytics.invoked=!0;analytics.methods=["trackSubmit","trackClick","trackLink","trackForm","pageview","identify","reset","group","track","ready","alias","debug","page","once","off","on","addSourceMiddleware","addIntegrationMiddleware","setAnonymousId","addDestinationMiddleware"];analytics.factory=function(e){return function(){var t=Array.prototype.slice.call(arguments);t.unshift(e);analytics.push(t);return analytics}};for(var e=0;e<analytics.methods.length;e++){var key=analytics.methods[e];analytics[key]=analytics.factory(key)}analytics.load=function(key,e){var t=document.createElement("script");t.type="text/javascript";t.async=!0;t.src="https://cdn.segment.com/analytics.js/v1/" + key + "/analytics.min.js";var n=document.getElementsByTagName("script")[0];n.parentNode.insertBefore(t,n);analytics._loadOptions=e};analytics._writeKey="ekoKjfMX6Le883YzBbThlr5Kpf4RnCdA";analytics.SNIPPET_VERSION="4.13.2";
	analytics.load("ekoKjfMX6Le883YzBbThlr5Kpf4RnCdA");
	analytics.page();
	}}();
  </script>
<script async src='/cdn-cgi/bm/cv/669835187/api.js'></script></head>
<body>
<header class="header">
<div class="header-wrapper">
<ul>
<li>
<a href="/">
<img src="/ai-notes/assets/images/layout/deeplearning.png">
</a>
</li>
<li> <a href="/ai-notes/" class="backToBlog">AI Notes</a></li>
<li class="header-nav-article"><a href="/ai-notes/initialization">Initialization</a></li>
<li class="header-nav-article"><a href="/ai-notes/optimization">Optimization</a></li>
</ul>
</div>
</header>
<div class="main">
<div class="container article-banner">
<div class="article-banner-content" id="vis-background">
<div id="cppn-overlay"></div>
</div>
<div>
<a class="cppn-control-toggle">
<div class="cppn-control">
<i class="fas fa-sliders-h"></i>
</div>
</a>
<div class="cppn-control" style="display: none">
<a class="cppn-control-toggle fa fa-times"></a>
<p>Network Depth:</p>
<label class="radio-container">
Shallow
<input type="radio" name="depth" value="3" checked />
<span class="checkmark"></span>
</label>
<label class="radio-container">
Deep
<input type="radio" name="depth" value="4" />
<span class="checkmark"></span>
</label>
<p>Layer Complexity:</p>
<label class="radio-container">
Simple
<input type="radio" name="complexity" value="20" checked />
<span class="checkmark"></span>
</label>
<label class="radio-container">
Complex
<input type="radio" name="complexity" value="25" />
<span class="checkmark"></span>
</label>
<p>Nonlinearity:</p>
<select id="activation" class="select-containter">
<option value="sin" selected>Sine</option>
<option value="cos">Cosine</option>
<option value="tanh">Tanh</option>
<option value="linear">Linear</option>
<option value="step">Step</option>
<option value="relu">Relu</option>
<option value="leakyRelu">Leaky Relu</option>
</select>
</div>
</div>
<script type="da8b9b91809b182b01d95155-text/javascript">

    $(".cppn-control-toggle").click(function() {
      $(".cppn-control").toggle();
    })

    var c1 = "255 139 34".split(" "),
        c2 = "255 87 87".split(" ");
        c3 = "255 31 103".split(" ");

    var cppn = cppnSetup([c1, c2, c3]),
        layers = 3,
        unit = 20
        activation = "sin";

    $("input[name='depth']").on("change", function () {
      layers = parseInt($(this).val());
      cppn.update(architecture(layers, unit), activation)
    });

    $("input[name='complexity']").on("change", function () {
      unit = parseInt($(this).val());
      cppn.update(architecture(layers, unit), activation)
    });

    $("#activation").on("change", function() {
      activation = $(this).val();
      cppn.update(architecture(layers, unit), activation)
    });


    function architecture(layers, units) {
      var arr = [5];
      for (var i = 0; i < layers; i++) {
        arr.push(units);
      }
      arr.push(3);
      return arr;
    }
</script>
<div class="banner-title">
<h1>Initializing neural networks</h1>
<p>Initialization can have a significant impact on convergence in training deep neural networks. Simple initialization schemes have been found to accelerate training, but they require some care to avoid common pitfalls. In this post, we'll explain how to initialize neural network parameters effectively.</p>
</div>
</div>
<div class="tableOfContent">
<p>TABLE OF CONTENTS</p>
<ul id="toc">
<li><span> I </span> <a href="#I">The importance of effective initialization</a></li>
<li><span> II </span> <a href="#II">The problem of exploding or vanishing gradients</a></li>
<li><span> III </span> <a href="#III">What is proper initialization?</a></li>
<li><span> IV </span> <a href="#IV">Mathematical justification for Xavier initialization</a></li>
</ul>
</div>
<section class="article-content">
<h1 id="I">I   The importance of effective initialization</h1>
<p>To build a machine learning algorithm, usually you’d define an architecture (e.g. Logistic regression, Support Vector Machine, Neural Network) and train it to learn parameters. Here is a common training process for neural networks:</p>
<ol>
<li>Initialize the parameters</li>
<li>Choose an <span class="sidenote">optimization algorithm</span></li>
<li>Repeat these steps:
<ol>
<li>Forward propagate an input</li>
<li>Compute the cost function</li>
<li>Compute the gradients of the cost with respect to parameters using backpropagation</li>
<li>Update each parameter using the gradients, according to the optimization algorithm</li>
</ol>
</li>
</ol>
<p>Then, given a new data point, you can use the model to predict its class.</p>
<p>The initialization step can be critical to the model’s ultimate performance, and it requires the right method. To illustrate this, consider the three-layer neural network below. You can try initializing this network with different methods and observe the impact on the learning.</p>
<div class="visualization hide-backToTop" id="playground">
<div class="visualization-column-1">
<h3>1. Choose input dataset</h3>
<p>Select a training dataset.</p>
<div id="playground_dataset"></div>
<p>This legend details the color scheme for labels, and the values of the weights/gradients.</p>
<div id="playground_legend"></div>
</div>
<div class="visualization-column-2">
<h3>2. Choose initialization method</h3>
<p>Select an initialization method for the values of your neural network parameters<sup class="footnote"></sup>.</p>
<label class="radio-container">Zero
<input type="radio" value="0" name="playground_init" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Too small
<input type="radio" value="0.01" name="playground_init" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Appropriate
<input type="radio" value="1" name="playground_init" checked="" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Too large
<input type="radio" value="100" name="playground_init" />
<span class="checkmark"></span>
</label>
<div id="playground_network"></div>
<p>Select whether to visualize the weights or gradients of the network above.</p>
<label class="radio-container">Weight
<input type="radio" value="weight" name="playground_link" checked="" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Gradient
 <input type="radio" value="gradient" name="playground_link" />
<span class="checkmark"></span>
</label>
</div>
<div class="visualization-column-1" style="min-height: 530px;">
<h3>3. Train the network.</h3>
<p>Observe the cost function and the decision boundary.</p>
<button class="button-transport" id="playground_reset" title="reset"><img src="../assets/images/layout/reset.png" /></button>
<button class="button-transport inactive" id="playground_start" title="start"><img src="../assets/images/layout/play.png" /></button>
<button class="button-transport hidden" id="playground_stop" title="stop"><img src="../assets/images/layout/pause.png" /></button>
<button class="button-transport inactive" id="playground_step" title="step"><img src="../assets/images/layout/fastforward.png" /></button>
<div class="line-break-sm"></div>
<div id="playground_loss"></div>
<div class="line-break-sm"></div>
<div id="playground_pred"></div>
</div>
</div>
<p>What do you notice about the gradients and weights when the initialization method is zero?</p>
<blockquote>
<p>Initializing all the weights with zeros leads the neurons to learn the same features during training.</p>
</blockquote>
<p>In fact, any constant initialization scheme will perform very poorly. Consider a <span class="sidenote">neural network</span> with two hidden units, and assume we initialize all the biases to 0 and the weights with some constant $\alpha$. If we forward propagate an input $(x_1,x_2)$ in this network, the output of both hidden units will be $relu(\alpha x_1 + \alpha x_2)$. Thus, both hidden units will have identical influence on the cost, which will lead to identical gradients. Thus, both neurons will evolve symmetrically throughout training, effectively preventing different neurons from learning different things.</p>
<p>What do you notice about the cost plot when you initialize weights with values too small or too large?</p>
<blockquote>
<p>Despite breaking the symmetry, initializing the weights with values (i) too small or (ii) too large leads respectively to (i) slow learning or (ii) divergence.</p>
</blockquote>
<p>Choosing proper values for initialization is necessary for efficient training. We will investigate this further in the next section.</p>
<h1 id="II">II   The problem of exploding or vanishing gradients</h1>
<p>Consider this 9-layer neural network.</p>
<p><img src="../assets/images/article/initialization/9layer.png" alt="9 layer" title="9 layer" /></p>
<p>At every iteration of the optimization loop (forward, cost, backward, update), we observe that backpropagated gradients are either amplified or minimized as you move from the output layer towards the input layer. This result makes sense if you consider the following example.</p>
<p>Assume all the activation functions are linear (identity function). Then the output activation is:</p>
<div class="kdmath">$$
\hat{y} = a^{[L]} = W^{[L]}W^{[L-1]}W^{[L-2]}\dots W^{[3]}W^{[2]}W^{[1]}x
$$</div>
<p>where $L=10$ and $W^{[1]},W^{[2]},\dots,W^{[L-1]}$ are all matrices of size $(2,2)$ because layers $[1]$ to $[L-1]$ have 2 neurons and receive 2 inputs. With this in mind, and for illustrative purposes, if we assume $W^{[1]} = W^{[2]} = \dots = W^{[L-1]} = W$ the output prediction is $\hat{y} = W^{[L]}W^{L-1}x$ (where $W^{L-1}$ takes the matrix $W$ to the power of $L-1$, while $W^{[L]}$ denotes the $L^{th}$ matrix).</p>
<p>What would be the outcome of initialization values that were too small, too large or appropriate?</p>
<h3 id="case-1-a-too-large-initialization-leads-to-exploding-gradients">Case 1: A too-large initialization leads to exploding gradients</h3>
<p>Consider the case where every weight is initialized slightly larger than the identity matrix.</p>
<div class="kdmath">$$
W^{[1]} = W^{[2]} = \dots = W^{[L-1]}=\begin{bmatrix}1.5 & 0 \\ 0 & 1.5\end{bmatrix}
$$</div>
<p>This simplifies to $\hat{y} = W^{[L]}1.5^{L-1}x$, and the values of $a^{[l]}$ increase exponentially with $l$. When these activations are used in backward propagation, this leads to the exploding gradient problem. That is, the gradients of the cost with the respect to the parameters are too big. This leads the cost to oscillate around its minimum value.</p>
<h3 id="case-2-a-too-small-initialization-leads-to-vanishing-gradients">Case 2: A too-small initialization leads to vanishing gradients</h3>
<p>Similarly, consider the case where every weight is initialized slightly smaller than the identity matrix.</p>
<div class="kdmath">$$
W^{[1]} = W^{[2]} = \dots = W^{[L-1]}=\begin{bmatrix}0.5 & 0 \\ 0 & 0.5\end{bmatrix}
$$</div>
<p>This simplifies to $\hat{y} = W^{[L]}0.5^{L-1}x$, and the values of the activation $a^{[l]}$ decrease exponentially with $l$. When these activations are used in backward propagation, this leads to the vanishing gradient problem. The gradients of the cost with respect to the parameters are too small, leading to convergence of the cost before it has reached the minimum value.</p>
<p>All in all, initializing weights with inappropriate values will lead to divergence or a slow-down in the training of your neural network. Although we illustrated the exploding/vanishing gradient problem with simple symmetrical weight matrices, the observation generalizes to any initialization values that are too small or too large.</p>
<h1 id="III">III   How to find appropriate initialization values</h1>
<p>To prevent the gradients of the network’s activations from vanishing or exploding, we will stick to the following rules of thumb:</p>
<ol>
<li>The <span class="sidenote">mean</span> of the activations should be zero.</li>
<li>The <span class="sidenote">variance</span> of the activations should stay the same across every layer.</li>
</ol>
<p>Under these two assumptions, the backpropagated gradient signal should not be multiplied by values too small or too large in any layer. It should travel to the input layer without exploding or vanishing.</p>
<p>More concretely, consider a <span class="sidenote">layer $l$</span>. Its forward propagation is:</p>
<div class="kdmath">$$
\begin{aligned}a^{[l-1]} &= g^{[l-1]}(z^{[l-1]})\\ z^{[l]} &= W^{[l]}a^{[l-1]} + b^{[l]}\\ a^{[l]} &= g^{[l]}(z^{[l]})\end{aligned}
$$</div>
<p>We would like the following to hold<sup class="footnote"></sup>:</p>
<div class="kdmath">$$
\begin{aligned}E[a^{[l-1]}] &= E[a^{[l]}]\\ Var(a^{[l-1]}) &= Var(a^{[l]})\end{aligned}
$$</div>
<p>Ensuring zero-mean and maintaining the value of the variance of the input of every layer guarantees no exploding/vanishing signal, as we’ll explain in a moment. This method applies both to the forward propagation (for activations) and backward propagation (for gradients of the cost with respect to activations). The recommended initialization is Xavier initialization (or one of its derived methods), for every layer $l$:</p>
<div class="kdmath">$$
\begin{aligned}W^{[l]} &\sim \mathcal{N}(\mu=0,\sigma^2 = \frac{1}{n^{[l-1]}})\\ b^{[l]} &= 0\end{aligned}
$$</div>
<p>In other words, all the weights of layer $l$ are picked randomly from a <span class="sidenote">normal distribution</span> with mean $\mu = 0$ and variance $\sigma^2 = \frac{1}{n^{[l-1]}}$ where $n^{[l-1]}$ is the number of neuron in layer $l-1$. Biases are initialized with zeros.</p>
<p>The visualization below illustrates the influence of the Xavier initialization on each layer’s activations for a five-layer fully-connected neural network.</p>
<div class="visualization hide-backToTop" id="mnist">
<div class="visualization-column-1">
<h3>1. Load your dataset</h3>
<p>Load 10,000 handwritten digits images (<a href="http://yann.lecun.com/exdb/mnist/">MNIST</a>).</p>
<button class="button emphasized" id="mnist_load">
Load MNIST (<span id="percent">0%</span>)
</button>
</div>
<div class="visualization-column-2">
<h3>2. Select an initialization method</h3>
<p>Among the below distributions, select the one to use to initialize your parameters<sup class="footnote"></sup>.</p>
<label class="radio-container">Zero
<input type="radio" value="zero" name="mnist_init" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Uniform
<input type="radio" value="uniform" name="mnist_init" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Xavier
<input type="radio" value="xe" name="mnist_init" checked="" />
<span class="checkmark"></span>
</label>
<label class="radio-container">Standard Normal
<input type="radio" value="normal" name="mnist_init" />
<span class="checkmark"></span>
</label>
</div>
<div class="visualization-column-1">
<h3>3. Train the network and observe</h3>
<p>The grid below refers to the input images, <span class="correct bold">Blue</span> squares represent correctly classified images. <span class="incorrect bold">Red</span> squares represent misclassified images.</p>
<button class="button-transport" id="mnist_reset"><img src="../assets/images/layout/reset.png" /></button>
<button class="button-transport inactive" id="mnist_start"><img src="../assets/images/layout/play.png" /></button>
<button class="button-transport hidden" id="mnist_stop"><img src="../assets/images/layout/pause.png" /></button>
<button class="button-transport inactive" id="mnist_step"><img src="../assets/images/layout/fastforward.png" /></button>
</div>
<div class="visualization-column-1">
<p>Input batch of 100 images</p>
<div id="mnist_input"></div>
<label class="viz">
Batch: <span id="batch">0</span> </label>
<label>
Epoch: <span id="epoch">0</span>
</label>
</div>
<div class="visualization-column-2">
<div id="mnist_network"></div>
</div>
<div class="visualization-column-1">
<p>Output predictions of 100 images</p>
<div id="mnist_output"></div>
<label class="viz">
Misclassified: <span id="accuracy">0/100</span>
</label>
<label>
Cost: <span id="cost">0.00</span>
</label>
</div>
<div class="visualization-column-full">
<img src="../assets/images/article/initialization/img4.png" class="viz-full-img" />
</div>
</div>
<p>You can find the theory behind this visualization in <a href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf?hc_location=ufi">Glorot et al. (2010)</a>. The next section presents the mathematical justification for Xavier initialization and explains more precisely why it is an effective initialization.</p>
<h1 id="IV">IV   Justification for Xavier initialization</h1>
<p>In this section, we will show that Xavier Initialization<sup class="footnote"></sup> keeps the variance the same across every layer. We will assume that our layer’s activations are normally distributed around zero. Sometimes it helps to understand the mathematical justification to grasp the concept, but you can understand the fundamental idea without the math.</p>
<p>Let’s work on the <span class="sidenote">layer $l$</span> described in part (III) and assume the activation function is <span class="sidenote">$tanh$</span>. The forward propagation is:</p>
<div class="kdmath">$$
\begin{aligned} z^{[l]} &= W^{[l]}a^{[l-1]} + b^{[l]} \\ a^{[l]} &= tanh(z^{[l]}) \end{aligned}
$$</div>
<p>The goal is to derive a relationship between $Var(a^{[l-1]})$ and $Var(a^{[l]})$. We will then understand how we should initialize our weights such that: $Var(a^{[l-1]}) = Var(a^{[l]})$.</p>
<p>Assume we initialized our network with appropriate values and the input is normalized. Early on in the training, we are in the <span class="sidenote">linear regime</span> of $tanh$. Values are small enough and thus $tanh(z^{[l]})\approx z^{[l]}$,<sup class="footnote"></sup> meaning that:</p>
<div class="kdmath">$$
Var(a^{[l]}) = Var(z^{[l]})
$$</div>
<p>Moreover, $z^{[l]} = W^{[l]}a^{[l-1]} + b^{[l]} = vector(z_1^{[l]},z_2^{[l]},\dots,z_{n^{[l]}}^{[l]})$ where $z_k^{[l]} = \sum_{j=1}^{n^{[l-1]}}w_{kj}^{[l]}a_j^{[l-1]} + b_k^{[l]}$. For simplicity, let’s assume that $b^{[l]} = 0$ (it will end up being true given the choice of initialization we will choose). Thus, looking <span class="sidenote">element-wise</span> at the previous equation $Var(a^{[l-1]}) = Var(a^{[l]})$ now gives:</p>
<div class="kdmath">$$
Var(a_k^{[l]}) = Var(z_k^{[l]}) = Var(\sum_{j=1}^{n^{[l-1]}}w_{kj}^{[l]}a_j^{[l-1]})
$$</div>
<p>A common math trick is to extract the summation outside the variance. To do this, we must make the following three <span class="sidenote">assumptions</span><sup class="footnote"></sup>:</p>
<ol>
<li>Weights are independent and identically distributed</li>
<li>Inputs are independent and identically distributed</li>
<li>Weights and inputs are mutually independent</li>
</ol>
<p>Thus, now we have:</p>
<div class="kdmath">$$
Var(a_k^{[l]}) = Var(z_k^{[l]}) = Var(\sum_{j=1}^{n^{[l-1]}}w_{kj}^{[l]}a_j^{[l-1]}) = \sum_{j=1}^{n^{[l-1]}}Var(w_{kj}^{[l]}a_j^{[l-1]})
$$</div>
<p>Another common math trick is to convert the variance of a product into a product of variances. Here is the <span class="sidenote">formula</span> for it:</p>
<div class="kdmath">$$
Var(XY) = E[X]^2Var(Y) + Var(X)E[Y]^2 + Var(X)Var(Y)
$$</div>
<p>Using this formula with $X = w_{kj}^{[l]}$ and $Y = a_j^{[l-1]}$, we get:</p>
<div class="kdmath">$$
Var(w_{kj}^{[l]}a_j^{[l-1]}) = E[w_{kj}^{[l]}]^2Var(a_j^{[l-1]}) + Var(w_{kj}^{[l]})E[a_j^{[l-1]}]^2 + Var(w_{kj}^{[l]})Var(a_j^{[l-1]})
$$</div>
<p>We’re almost done! The first assumption leads to $E[w_{kj}^{[l]}]^2 = 0$ and the second assumption leads to $E[a_j^{[l-1]}]^2 = 0$ because weights are initialized with zero mean, and inputs are normalized. Thus:</p>
<div class="kdmath">$$
Var(z_k^{[l]}) = \sum_{j=1}^{n^{[l-1]}}Var(w_{kj}^{[l]})Var(a_j^{[l-1]}) = \sum_{j=1}^{n^{[l-1]}}Var(W^{[l]})Var(a^{[l-1]}) = n^{[l-1]}Var(W^{[l]})Var(a^{[l-1]})
$$</div>
<p>The equality above results from our first assumption stating that:</p>
<div class="kdmath">$$
Var(w_{kj}^{[l]}) = Var(w_{11}^{[l]}) = Var(w_{12}^{[l]})=\dots = Var(W^{[l]})
$$</div>
<p>Similarly the second assumption leads to:</p>
<div class="kdmath">$$
Var(a_j^{[l-1]}) = Var(a_1^{[l-1]}) = Var(a_2^{[l-1]})=\dots = Var(a^{[l-1]})
$$</div>
<p>With the same idea:</p>
<div class="kdmath">$$
Var(z^{[l]}) = Var(z_k^{[l]})
$$</div>
<p>Wrapping up everything, we have:</p>
<div class="kdmath">$$
Var(a^{[l]}) = n^{[l-1]}Var(W^{[l]})Var(a^{[l-1]})
$$</div>
<p>Voilà! If we want the variance to stay the same across layers ($Var(a^{[l]}) = Var(a^{[l-1]})$), we need $Var(W^{[l]}) = \frac{1}{n^{[l-1]}}$. This justifies the choice of variance for Xavier initialization.</p>
<p>Notice that in the previous steps we did not choose a specific layer $l$. Thus, we have shown that this expression holds for every layer of our network. Let $L$ be the output layer of our network. Using this expression at every layer, we can link the output layer’s variance to the input layer’s variance:</p>
<div class="kdmath">$$
\begin{aligned} Var(a^{[L]}) &= n^{[L-1]}Var(W^{[L]})Var(a^{[L-1]}) \\ &= n^{[L-1]}Var(W^{[L]})n^{[L-2]}Var(W^{[L-1]})Var(a^{[L-2]})\\ &=\dots\\ &= \left[\prod_{l=1}^L n^{[l-1]}Var(W^{[l]})\right]Var(x)\end{aligned}
$$</div>
<p>Depending on how we initialize our weights, the relationship between the variance of our output and input will vary dramatically. Notice the following three cases.</p>
<div class="kdmath">$$
n^{[l-1]}Var(W^{[l]}) \begin{cases} < 1 &\implies \text{Vanishing Signal}\\ = 1 & \implies Var(a^{[L]}) = Var(x)\\ > 1 & \implies \text{Exploding Signal}\end{cases}
$$</div>
<p>Thus, in order to avoid the vanishing or exploding of the forward propagated signal, we must set $n^{[l-1]}Var(W^{[l]}) = 1$ by initializing $Var(W^{[l]}) = \frac{1}{n^{[l-1]}}$.</p>
<p>Throughout the justification, we worked on activations computed during the forward propagation. The same result can be derived for the backpropagated gradients. Doing so, you will see that in order to avoid the vanishing or exploding gradient problem, we must set $n^{[l]}Var(W^{[l]}) = 1$ by initializing $Var(W^{[l]}) = \frac{1}{n^{[l]}}$.</p>
<h1 id="conclusion">Conclusion</h1>
<p>In practice, Machine Learning Engineers using Xavier initialization would either initialize the weights as $\mathcal{N}(0,\frac{1}{n^{[l-1]}})$ or as $\mathcal{N}(0,\frac{2}{n^{[l-1]} + n^{[l]}})$. The variance term of the latter distribution is the harmonic mean of $\frac{1}{n^{[l-1]}}$ and $\frac{1}{n^{[l]}}$.</p>
<p>This is a theoretical justification for Xavier initialization. Xavier initialization works with tanh activations. Myriad other initialization methods exist. If you are using ReLU, for example, a common initialization is He initialization (<a href="https://arxiv.org/pdf/1502.01852.pdf">He et al., Delving Deep into Rectifiers</a>), in which the weights are initialized by multiplying by 2 the variance of the Xavier initialization. While the justification for this initialization is slightly more complicated, it follows the same thought process as the one for tanh.</p>
<div class="column-6-8 column-align margin cta">
<p><strong>Learn more about how to effectively initialize parameters in</strong><br /><strong>Course 2 of the Deep Learning Specialization</strong></p>
<p><a target="_blank" href="https://bit.ly/2VMCWcR" class="button-transport">Enroll now</a></p>
</div>
<div class="sidenote-body">
<p class="caption">Examples include Adam, Momentum, RMSProp, Stochastic and Batch Gradient Descent methods.</p>
</div>
<div class="sidenote-body">
<p class="caption">A neural network with two hidden relu units and a sigmoid output unit.</p>
</div>
<div class="sidenote-body">
<p class="caption">Mean is a measure of the center or expectation of a random variable.</p>
</div>
<div class="sidenote-body">
<p class="caption">Variance is a measure of how much a random variable is spread around its mean. In deep learning, the random variable could be the data, the prediction, the weights, the activations, etc.</p>
</div>
<div class="sidenote-body">
<p class="caption"><img src='../assets/images/article/initialization/layerl.png'> $a^{[l-1]}$ represents the input to layer $l$ and $a^{[l]}$ represents the output. $g^{[l]}$ is the activation function of layer $l$. $n^{[l]}$ is the number of neuron in layer $l$.</p>
</div>
<div class="sidenote-body">
<p class="caption"><img src='../assets/images/article/initialization/normal.png'> Values generated from a normal distribution $\mathcal{N}(\mu,\sigma^2)$ are symmetric around the mean $\mu$.</p>
</div>
<div class="sidenote-body">
<p class="caption"><img src='../assets/images/article/initialization/layerl.png'> $a^{[l-1]}$ represents the input to layer $l$ and $a^{[l]}$ represents the output.</p>
</div>
<div class="sidenote-body">
<p class="caption">$tanh$ is a non-linear function defined as $tanh(x) = \frac{1 - e^{-2x}}{1 + e^{-2x}}$.</p>
</div>
<div class="sidenote-body">
<p class="caption">Important properties of $tanh$ are its parity ($tanh(-x) = -tanh(x)$) and its linearity around 0 ($tanh'(0) = 1$).</p>
</div>
<div class="sidenote-body">
<p class="caption">The variance of the vector is the same as the variance of any of its entries, because all its entries are drawn independently and identically from the same distribution (i.i.d.).</p>
</div>
<div class="sidenote-body">
<p class="caption">These assumptions are not always true, but they are necessary to approach the problem theoretically at this point.</p>
</div>
<div class="sidenote-body">
<p class="caption">This is only true for independent random variables.</p>
</div>
</section>
</div>
<div class="foot-note">
<div class="foot-note-header">
<h4 class="reference">Authors</h4>
</div>
<div class="foot-note-content">
<ol class="reference ">
<li><a href="https://twitter.com/kiankatan">Kian Katanforoosh</a> - Written content and structure.</li>
<li><a href="http://daniel-kunin.com">Daniel Kunin</a> - Visualizations (created using <a href="https://d3js.org/">D3.js</a> and <a href="https://js.tensorflow.org/">TensorFlow.js</a>).</li>
</ol>
</div>
<div class="foot-note-header">
<h4 class="reference">Acknowledgments</h4>
</div>
<div class="foot-note-content">
<ol class="reference">
<li>The template for the article was designed by <a href="https://www.jingru-guo.com/">Jingru Guo</a> and inspired by <a href="https://distill.pub/">Distill</a>.</li>
<li>The first visualization adapted code from Mike Bostock's <a href="https://bl.ocks.org/mbostock/f48ff9c1af4d637c9a518727f5fdfef5">visualization</a> of the Goldstein-Price function.</li>
<li>The banner visualization adapted code from deeplearn.js's implementation of a <a href="https://en.wikipedia.org/wiki/Compositional_pattern-producing_network">CPPN</a>.</li>
</ol>
</div>
<div class="foot-note-header">
<h4 class="reference">Footnotes</h4>
</div>
<div class="foot-note-content">
<ol class="reference" id="fn">
<li class="footnote-body"><span>All bias parameters are initialized to zero and weight parameters are drawn from a normal distribution with zero mean and selected variance.</span></li>
<li class="footnote-body"><span>Under the hypothesis that all entries of the weight matrix $W^{[l]}$ are picked from the same distribution, $Var(w_{11}) = Var(w_{12}) = \dots = Var(w_{n^{[l]}n^{[l-1]}})$. Thus, $Var(W^{[l]})$ indicates the variance of any entry of $W^{[l]}$ (they're all the same!). Similarly, we will denote $Var(x)$ (resp. $Var(a^{[l]})$) the variance of any entry of $x$ (resp. $a^{[l]}$). It is a fair approximation to consider that every pixel of a "real-world image" $x$ is distributed according to the same distribution.</span></li>
<li class="footnote-body"><span>All bias parameters are initialized to zero and weight parameters are drawn from either "Zero" distribution ($w_{ij} = 0$), "Uniform" distribution ($w_{ij} \sim U(\frac{-1}{\sqrt{n^{[l-1]}}},\frac{1}{\sqrt{n^{[l-1]}}})$), "Xavier" distribution ($w_{ij} \sim N(0,\frac{1}{\sqrt{n^{[l-1]}}})$), or "Standard Normal" distribution ($w_{ij} \sim N(0,1)$).</span></li>
<li class="footnote-body"><span>Concretely it means we pick every weight randomly and independently from a normal distribution centered in $\mu = 0$ and with variance $\sigma^2 = \frac{1}{n^{[l-1]}}$.</span></li>
<li class="footnote-body"><span>We assume that $W^{[l]}$ is initialized with small values and $b^{[l]}$ is initialized with zeros. Hence, $Z^{[l]} = W^{[l]}A^{[l-1]} + b^{[l]}$ is small and we are in the linear regime of $tanh$. Remember the slope of $tanh$ around zero is one, thus $tanh(Z^{[l]}) \approx Z^{[l]}$.</span></li>
<li class="footnote-body"><span>The first assumption will end up being true given our initialization scheme (we pick weights randomly according to a normal distribution centered at zero). The second assumption is not always true. For instance in images, inputs are pixel values, and pixel values in the same region are highly correlated with each other. On average, it’s more likely that a green pixel is surrounded by green pixels than by any other pixel color, because this pixel might be representing a grass field, or a green object. Although it’s not always true, we assume that inputs are distributed identically (let’s say from a normal distribution centered at zero.) The third assumption is generally true at initialization, given that our initialization scheme makes our weights independent and identically distributed (i.i.d.).</span></li>
</ol>
</div>
<div class="foot-note-header">
<h4 class="reference">Reference</h4>
</div>
<div class="foot-note-content">
<p class="reference">To reference this article in an academic context, please cite this work as:</p>
<p class="citation">Katanforoosh & Kunin, "Initializing neural networks", deeplearning.ai, 2019.</p>
</div>
</div>
<div class="footer-generic hide-backToTop">
<div class="container">
<p class="footer-note">
© Deeplearning.ai 2021</br>
<a href="/privacy/">PRIVACY POLICY</a> <a href="/terms-of-use/">TERMS OF USE</a>
</p>
</div>
</div>
<div class="backToTop">
<p>↑ Back to top</p>
</div>
<link rel="stylesheet" href="/ai-notes/assets/css/article/initialization/playground.css">
<link rel="stylesheet" href="/ai-notes/assets/css/article/initialization/mnist.css">
<script src="/ai-notes/assets/js/article/initialization/playground/data.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/playground/nn.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/playground/viz.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/mnist/nn.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/mnist/zip/zip.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/mnist/zip/zip-ext.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/mnist/data.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="/ai-notes/assets/js/article/initialization/mnist/viz.js" type="da8b9b91809b182b01d95155-text/javascript"></script>
<script src="https://ajax.cloudflare.com/cdn-cgi/scripts/7d0fa10a/cloudflare-static/rocket-loader.min.js" data-cf-settings="da8b9b91809b182b01d95155-|49" defer=""></script><script type="text/javascript">(function(){window['__CF$cv$params']={r:'6587860d8dbcdcc2',m:'4cc4031177199faccb8ea41e50834fda7e245dd8-1622540879-1800-AYyb3Zl9rvVBjSxB1DENiJWp+E8h6x5vhljchsTxj6GCXGHbvkeyR3lB/fsGLvJvbOqNqKohsRSa8nNcGKoivHrRt2G0Zxzj4lj50KOY5jEOCgDd7vs4lzciJ69yuh/yga/j6Ww8+ImwAs6Nz4XyPQo=',s:[0xd60ad15ec1,0xffaf7c16eb],}})();</script><script defer src="https://static.cloudflareinsights.com/beacon.min.js" data-cf-beacon='{"rayId":"6587860d8dbcdcc2","token":"4c1c83ca2dd644dea02182b686a741bd","version":"2021.5.2","si":10}'></script>
</body>
</html>