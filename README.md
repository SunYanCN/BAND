# BAND：BERT Application aNd Deployment

A simple and efficient BERT model training and deployment framework.

<!-- PROJECT SHIELDS -->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/SunYanCN/BAND">
    <img src="figures/logo.png" alt="Logo" width="100" height="100">
  </a>

  <h3 align="center">BAND</h3>
  <p align="center">
    BAND：BERT Application aNd Deployment
    <br />
    <a href="https://sunyancn.github.io/BAND/"><strong>探索本项目的文档 »</strong></a>
    <br />
    <br />
    <a href="https://github.com/SunYanCN/BAND/tree/master/examples">查看Demo</a>
    ·
    <a href="https://github.com/SunYanCN/BAND/issues/new?assignees=&labels=&template=bug_report.md&title=">报告Bug</a>
    ·
    <a href="https://github.com/SunYanCN/BAND/issues/new?assignees=&labels=&template=feature_request.md&title=">提出新特性</a>
        ·
    <a href="https://github.com/SunYanCN/BAND/issues/new?assignees=&labels=&template=custom.md&title=">问题交流</a>
  </p>

</p>

<h2 align="center">What is it</h3>  
  
**Encoding/Embedding** is a upstream task of encoding any inputs in the form of text, image, audio, video, transactional data to fixed length vector. Embeddings are quite popular in the field of NLP, there has been various Embeddings models being proposed in recent years by researchers, some of the famous one are bert, xlnet, word2vec etc. The goal of this repo is to build one stop solution for all embeddings techniques available, here we are starting with popular text embeddings for now and later on we aim  to add as much technique for image, audio, video inputs also.  
**Finally**, **`embedding-as-service`** help you to encode any given text to fixed length vector from supported embeddings and models.  
  
<h2 align="center">💾 Installation</h2>  
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>
  
  
Install the band via `pip`.   
```bash  
$ pip install band -U
```  
> Note that the code MUST be running on **Python >= 3.6**. Again module does not support Python 2!  
  
<h2 align="center">⚡ ️Getting Started</h2> 
<p align="right"><a href="#embedding-as-service"><sup>▴ Back to top</sup></a></p>

## Dataset 
For more information about dataset, see

| Dataset Name | Language |             TASK              |        Description         |
| :----------: | :------: | :---------------------------: | :------------------------: |
| ChnSentiCorp |    CN    |      Text Classification      |   Binary Classification    |
|    LCQMC     |    CN    |     Question Answer Match     |   Binary Classification    |
|   MSRA_NER   |    CN    |   Named Entity Recognition    |     Sequence Labeling      |
|    Toxic     |    EN    |      Text Classification      |  Multi-label Multi-label   |
|   Thucnews   |    CN    |      Text Classification      | Multi-class Classification |
|    SQUAD     |    EN    | Machine Reading Comprehension |            Span            |
|     DRCD     |    CN    | Machine Reading Comprehension |            Span            |
|     CMRC     |    CN    | Machine Reading Comprehension |            Span            |
|     GLUE     |    EN    |                               |                            |

## Current Pretrained Models
For more information about pretrained models, see
<!-- links -->
[your-project-path]: SunYanCN/BAND
[contributors-shield]: https://img.shields.io/github/contributors/SunYanCN/BAND.svg?style=flat-square
[contributors-url]: https://github.com/SunYanCN/BAND/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/SunYanCN/BAND.svg?style=flat-square
[forks-url]: https://github.com/SunYanCN/BAND/network/members
[stars-shield]: https://img.shields.io/github/stars/SunYanCN/BAND.svg?style=flat-square
[stars-url]: https://github.com/SunYanCN/BAND/stargazers
[issues-shield]: https://img.shields.io/github/issues/SunYanCN/BAND.svg?style=flat-square
[issues-url]: https://github.com/SunYanCN/BAND/issues
[license-shield]: https://img.shields.io/github/license/SunYanCN/BAND.svg?style=flat-square
[license-url]: https://github.com/SunYanCN/BAND/blob/master/LICENSE

## Stargazers over time

[![Stargazers over time](https://starchart.cc/SunYanCN/BAND.svg)](https://starchart.cc/SunYanCN/BAND)