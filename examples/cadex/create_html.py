#!/usr/bin/python3

import argparse
parser = argparse.ArgumentParser(description='Create Interactive SVG')
parser.add_argument('svgfile', type=str, help='input file name for svg')


args = parser.parse_args()
svgfile = args.svgfile

if ".svg" not in svgfile:
    print ("NOT svg file")
    quit()

html = """
<!--
    * Copyright (c) 2015 Mountainstorm
    * 
    * Permission is hereby granted, free of charge, to any person obtaining a copy
    * of this software and associated documentation files (the "Software"), to deal
    * in the Software without restriction, including without limitation the rights
    * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    * copies of the Software, and to permit persons to whom the Software is
    * furnished to do so, subject to the following conditions:
    * 
    * The above copyright notice and this permission notice shall be included in all
    * copies or substantial portions of the Software.
    * 
    * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF AERCHANTABILITY,
    * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEAENT. IN NO EVENT SHALL THE
    * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    * SOFTWARE.
  -->
<html>
  <head>
    <!--link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.4/css/bootstrap.min.css"-->
    <link rel="stylesheet" href="css/bootstrap.min.css">
    <link rel="stylesheet" href="css/graphviz.svg.css">
  </head>

  <style>
    #instructions {
    color: #000000;
    position: absolute;
    z-index: 100;
    bottom: 0px;
    left: 0px;
    }
    .div{
        overflow-x: auto;
    }
  </style>
  <body>
    Interactive Call Graph<p>
      <h4 id="instructions">Click node to highlight; Shift-scroll to zoom; Esc to unhighlight</h4>
      <div id="graph" style="width: 100%; height: 100%; overflow: scroll;"></div>

      <!--script type="text/javascript" src="https://code.jquery.com/jquery-2.1.3.min.js"></script-->
      <!--script type="text/javascript" src="https://cdn.rawgit.com/jquery/jquery-mousewheel/master/jquery.mousewheel.min.js"></script-->
      <!--script type="text/javascript" src="https://cdn.rawgit.com/jquery/jquery-color/master/jquery.color.js"></script-->
      <!--script type="text/javascript" src="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/3.3.4/js/bootstrap.min.js"></script-->
      <script type="text/javascript" src="js/jquery-2.1.3.min.js"></script>
      <script type="text/javascript" src="js/jquery.mousewheel.min.js"></script>
      <script type="text/javascript" src="js/jquery.color.js"></script>
      <script type="text/javascript" src="js/bootstrap.min.js"></script>
      <script type="text/javascript" src="js/jquery.graphviz.svg.js"></script>
      <script type="text/javascript">
        $(document).ready(function(){
        $("#graph").graphviz({
        url: "FILENAME.svg", 
        ready: function() {
        var gv = this
        gv.nodes().click(function () {
        var $set = $()
        $set.push(this)
        $set = $set.add(gv.linkedFrom(this, true))
        $set = $set.add(gv.linkedTo(this, true))
        gv.highlight($set, true)
        gv.bringToFront($set)
        })
        $(document).keydown(function (evt) {
        if (evt.keyCode == 27) {
        gv.highlight()
        }
        })
        }
        });
        });
      </script>
  </body>
</html>
"""

html = html.replace("FILENAME.svg",svgfile)

with open(svgfile.replace(".svg",".html"),"w") as f:
    f.write(html)
