# BAND：BERT Application aNd Deployment

A simple and efficient BERT model training and deployment framework，一个简单高效的 BERT 模型训练和部署框架

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
    <a href="https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/issues/new?assignees=&labels=&template=bug_report.md&title=">报告Bug</a>
    ·
    <a href="https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/issues/new?assignees=&labels=&template=feature_request.md&title=">提出新特性</a>
        ·
    <a href="https://github.com/SunYanCN/BERT-chinese-text-classification-and-deployment/issues/new?assignees=&labels=&template=custom.md&title=">问题交流</a>
  </p>

</p>

## Dataset 
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
