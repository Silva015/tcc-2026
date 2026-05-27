\NeedsTeXFormat{LaTeX2e}[1994/12/01]
\ProvidesClass{UnB-CIC-English}[2014/07/10 Class for UnB CIC works in English]

%%
%% --- Department contextualization ---
%%

\def\@departamento{Department of Computer Science}
\def\@instituto{Institute of Exact Sciences}
\def\@universidade{University of Bras\'{\i}lia}
\def\@siglauni{UnB}
\def\@cidade{Bras\'ilia}
\def\@uf{DF}
\def\@cidadeuf{\@cidade--\@uf}
\def\@pais{Brazil}

%%
%% --- Option definitions ---
%%
\RequirePackage{xkeyval}

%line spacing
\newif\if@singlespacing
\newif\if@onehalfspacing
\newif\if@baselineskip
\DeclareOptionX{singlespacing}{\@singlespacingtrue \@onehalfspacingfalse}
\DeclareOptionX{onehalfspacing}{\@onehalfspacingtrue \@singlespacingfalse}
\DeclareOptionX{doublespacing}{\@singlespacingfalse \@onehalfspacingfalse}%\else
\DeclareOptionX{baselineskip}{\@baselineskiptrue}

%course options
\newif\if@posgraduacao\@posgraduacaofalse%
\DeclareOptionX{bacharelado}{%
    \gdef\@nomecurso{Bachelor of Science in Computer Science}%
    \gdef\@mododeescrita{Thesis}}%

\DeclareOptionX{licenciatura}{%
    \gdef\@nomecurso{Computer Science Teaching Degree}%
    \gdef\@mododeescrita{Thesis}}%

\DeclareOptionX{engenharia}{%
    \gdef\@nomecurso{Computer Engineering Degree}%
    \gdef\@mododeescrita{Undergraduate thesis}}%


\def\@nomecurso{\@empty}%
\def\@assunto{Thesis (Undergraduate)}
\DeclareOptionX{mestrado}{\@posgraduacaotrue%
    \gdef\@nomecurso{Master's Degree}%
    \gdef\@assunto{Dissertation (Master's)}%
    \gdef\@mododeescrita{Dissertation}}%

\newif\if@doutorado\@doutoradofalse%
\DeclareOptionX{doutorado}{\@posgraduacaotrue\@doutoradotrue%
    \gdef\@nomecurso{Doctorate}%
    \gdef\@assunto{Doctoral Thesis}%
    \gdef\@mododeescrita{Thesis}}%

\DeclareOptionX{ppginf}{%
					\gdef\@nomeDoPrograma{Informatics}}%

\newif\if@ppca\@ppcafalse%
\DeclareOptionX{ppca}{\@ppcatrue%
					\gdef\@nomeDoPrograma{Applied Computing}%
					\gdef\@nomecurso{Professional Master's Degree}}%

\newif\if@qualificacao\@qualificacaofalse%
\DeclareOptionX{qualificacao}{\@qualificacaotrue}%

%PDF generation mode
\newif\if@ForScreenView
\newif\if@LoadHyperref
\DeclareOptionX{impressao}{\@ForScreenViewfalse\@LoadHyperreffalse}
\DeclareOptionX{hyperref}[load]{\@ForScreenViewtrue%
	\def\@tempa{#1}\def\@tempb{load}\def\@tempc{config}%
	\ifx\@tempa\@tempb\@LoadHyperreftrue%
	\else%
		\ifx\@tempa\@tempc\@LoadHyperreffalse%
		\else
			\ClassError{UnB-CIC-English}
				{Invalid value for hyperref option: #1}
				{Valid values: hyperref,hyperref=load,hyperref=config}
		\fi
	\fi%
}

%fonts
\def\fontelogo{\@ifundefined{XeTeXversion}{ua1}{Arial}}
\DeclareOptionX{10pt}{\gdef\@fontsize{10pt}}
\DeclareOptionX{11pt}{\gdef\@fontsize{11pt}}
\DeclareOptionX{12pt}{\gdef\@fontsize{12pt}}

%page styles
\DeclareOptionX{prestyle}{\gdef\style@pretextual{#1}}
\DeclareOptionX{textstyle}{\gdef\style@textual{#1}}
\DeclareOptionX{chapstyle}{\gdef\style@chapter{#1}}

%draft
\newif\if@draftthesis
\DeclareOptionX{draft}{\@draftthesistrue}

%%
%% --- Option processing and base class ---
%%

%pass unknown options
\DeclareOptionX*{\PassOptionsToClass{\CurrentOption}{report}}

%default options
\ExecuteOptionsX{onehalfspacing,hyperref,prestyle=plain,textstyle=plain,chapstyle=plain}
\ProcessOptionsX\relax

\newcommand{\ifhyperref}[2]{\if@ForScreenView #1\else #2\fi}

%define macro only if undefined
\def\@defifundef#1#2{\begingroup\expandafter\ifx\csname #1\endcsname\relax%
\expandafter\gdef\csname #1\endcsname{#2}
\fi\endgroup}

\if@draftthesis
	\PassOptionsToPackage{headheight=13pt,margin={2cm,1.5cm}}{geometry}
    \@singlespacingfalse \@onehalfspacingtrue %force onehalfspacing
    \@defifundef{@fontsize}{11pt}
\else
	\PassOptionsToPackage{headheight=15pt,vmargin={3cm,2cm}}{geometry}
	\if@ForScreenView
	\PassOptionsToPackage{hmargin={2.5cm,2.5cm}}{geometry}
    \else
	\PassOptionsToPackage{hmargin={3cm,2cm}}{geometry}
    \fi%
    \@defifundef{@fontsize}{12pt}
\fi

\LoadClass[\@fontsize]{report}

%%
%% --- Required packages ---
%%


\RequirePackage[american,brazil]{babel}% Languages
\RequirePackage[fixlanguage]{babelbib}%  Reference language
\RequirePackage[utf8]{inputenc}%         UTF-8 encoding
\RequirePackage[T1]{fontenc}%            Enable search with Latin characters
\RequirePackage{lmodern}%                Enable search with Latin characters
\RequirePackage{amsmath}%				 Equation environment
\RequirePackage{xcolor}% 				 Required for some unidentified package. Related to AtBeginDocument and \@conteudoPreTextual (if placed inside begin{document}, it works)
\RequirePackage[figure,table]{totalcount}%


\@ifundefined{XeTeXversion}{%
	\PassOptionsToPackage{pdftex}{hyperref}

	\RequirePackage{relsize} %used in \@logo
	\def\@textscale#1#2{\textscale{#1}{#2}}
}{
	\PassOptionsToPackage{xetex}{hyperref}

	\def\@textscale#1#2{{\addfontfeatures{Scale=#1}#2}}
}%

\RequirePackage{graphicx}\graphicspath{{.}{img/}}%
\RequirePackage{boites}
\RequirePackage[a4paper,includefoot]{geometry}

\if@baselineskip\else
	%define spacing via setspace
	\RequirePackage{setspace}
\fi

\PassOptionsToPackage{pdfdisplaydoctitle=true,pagebackref,linktocpage=true}{hyperref}%

\if@LoadHyperref
	\RequirePackage[hidelinks]{hyperref}%
\fi

\usepackage[acronym,nonumberlist]{glossaries}% After hyperref!
\makeglossaries%

%to generate the label {TotPaginas}
\RequirePackage{atbegshi}
\newcounter{TotPaginas}

%%
%% --- Base class adaptation ---
%%
\newcommand{\chapterformat}{%chapter title formatting
\Huge \bfseries
}

\newcommand{\chapteralign}{%chapter title alignment
	\ClassError{UnB-CIC-English}
			{Command \protect\pretextual\space was not called}
			{There cannot be titles before the \protect\pretextual\space command}
}

%support for configurable alignment and formatting
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \chapteralign \normalfont
    \ifnum \c@secnumdepth >\m@ne
        \huge\bfseries \@chapapp\space \thechapter
        \par\nobreak
        \vskip 20\p@
    \fi
    \interlinepenalty\@M
    \chapterformat #1\par\nobreak
    \vskip 40\p@
  }}

\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \chapteralign
    \normalfont
    \interlinepenalty\@M
    \chapterformat  #1\par\nobreak
    \vskip 40\p@
  }}

%remove forced \MakeUppercase in headers
\let\old@MakeUppercase\MakeUppercase
\let\old@markboth\markboth
\def\markboth#1#2{\def\MakeUppercase##1{##1}\old@markboth{#1}{#2}\let\MakeUppercase\old@MakeUppercase}
\let\old@markright\markright
\def\markright#1{\def\MakeUppercase##1{##1}\old@markright{#1}\let\MakeUppercase\old@MakeUppercase}

%avoid orphans/widows and footnote breaks
\clubpenalty=9999
\widowpenalty=9999
\interfootnotelinepenalty=5000
\raggedbottom

%bring list items closer
\def\changeitemsep{\setlength{\itemsep}{.5\itemsep}}
\let\old@enumerate=\enumerate
\def\enumerate{\old@enumerate%
\changeitemsep}
\let\old@itemize=\itemize
\def\itemize{\old@itemize%
\changeitemsep}

%line spacing for verse, quote, and quotation
%should be simple and with smaller font
\let\old@quote\quote
\let\old@endquote\endquote
\renewenvironment*{quote}
	{\old@quote\begin{singlespace}\small}
	{\end{singlespace}\old@endquote}

\let\old@quotation\quotation
\let\old@endquotation\endquotation
\renewenvironment*{quotation}
	{\old@quotation\begin{singlespace}\small}
	{\end{singlespace}\old@endquotation}

\let\old@verse\verse
\let\old@endverse\endverse
\renewenvironment*{verse}
	{\old@verse\begin{singlespace}\small}
	{\end{singlespace}\old@endverse}

%new definition environment
\newcommand{\definitioname}{Definition}
\newcounter{definition}
\newenvironment{definition}
	{\refstepcounter{definition}%
	\let\old@fboxsep\fboxsep\setlength{\fboxsep}{1em}%
	\vskip.5\fboxsep\breakbox%
	\noindent\textbf{\definitioname~\arabic{definition}\ \ }}
	{\endbreakbox\vskip.5\fboxsep%
	\setlength{\fboxsep}{\old@fboxsep}}


%automatic table figures in environment #1
\newcommand{\SetTableFigures}[1]{
	\expandafter\let\csname old@#1\expandafter\endcsname\csname #1\endcsname
	\expandafter\let\csname old@end#1\expandafter\endcsname\csname end#1\endcsname
	\renewenvironment{#1}
	{\begingroup\@tablefigures\csname old@#1\endcsname}
	{\csname old@end#1\endcsname\endgroup}
}

%add item to table of contents
\def\addtocchapter#1{
	\cleardoublepage
	\phantomsection %adjust hyperref reference
	\addcontentsline{toc}{chapter}{#1}
}

%configure references section
\expandafter\let\expandafter\@bibliography\csname bibliography\endcsname%
\renewcommand{\bibliography}[1]{%
	%rename section after evil Babel influence
	\renewcommand*{\bibname}{References}
	\addtocchapter{\bibname}
	\singlespacing\@bibliography{#1}\defaultspacing}

%maketitle to generate initial pages
\def\maketitle{%
	\if@ForScreenView\relax\else\capa\fi
	\pretextual\folharosto%
    \if@posgraduacao%
		\if@qualificacao%
		\else%
			\newpage\thispagestyle{empty}%
			\begin{picture}(0,0)\put(-90,-747){\includegraphics{doc/BDM.pdf}}\end{picture}%
			% \includepdf (appropriate command in this case) needs to be reconfigured in pretextual mode, \includegraphics is a quick solution, but uses arbitrary values to position the page.
			\clearpage%
		\fi%
	\fi%
	\newpage\folhaaprovacao%
}

%hyperref support for annexes
\ifhyperref{%
	\def\Hy@annexstring{annex}
	\let\HyOrg@anexos\anexos
	\def\anexos{%
		\gdef\theHchapter{\Roman{chapter}}%
		\xdef\Hy@chapapp{\Hy@annexstring}%
		\HyOrg@anexos
	}
}{}

%%
%% --- Auxiliary macros/commands ---
%%
%%
%% --- Auxiliary macros/commands ---
%%
\def\prof{\@ifnextchar[{\@prof@}{\@prof}}
\def\@prof#1{Prof.\ #1}
\def\@prof@[#1]#2{Prof.\ #2}

\def\dr{\@ifnextchar[{\@dr@}{\@dr}}
\def\@dr#1{Dr.\ #1}
\def\@dr@[#1]#2{Dr.\ #2}

%%

%%
%% --- Work pages ---
%%

\def\@lskipoffset#1{%line break according to reference
	\begingroup%
		\newskip\len@lskipoffset%spacing reference
		\setlength{\len@lskipoffset}{-.275\len@uadesch}%
		\addtolength{\len@lskipoffset}{#1\len@uacaph}%
		\\[\len@lskipoffset]%
	\endgroup%
}
\newcommand*{\@logosize}{20}%font size used (in pt)
\newcommand{\@logo}[1][positivo_cor]{%print UnB logo
	\begingroup%
		\newskip\len@brasiliaw%width of the word Brasilia
		\newskip\len@uadesch%URW Arial descender height
		\newskip\len@uacaph%URW Arial cap height
		%
		\begin{spacing}{0}%remove line spacing
        \@ifundefined{XeTeXversion}
	{\fontfamily{\fontelogo}}%
			{\fontspec{\fontelogo}}%
		\fontsize{\@logosize}{0}\bfseries\selectfont%
		\settodepth{\len@uadesch}{jgpy}\settoheight{\len@uacaph}{B}%desc & cap heights
		\settowidth{\len@brasiliaw}{Bras\'ilia}%
		\addtolength{\len@brasiliaw}{-.105em}%adjust small whitespace before B in this font
		\includegraphics[width=\len@brasiliaw]{#1}\\[.63\len@uacaph]%logo with the width of the word Brasilia
		\@universidade\@lskipoffset{.5}%
		\@textscale{.5}{\@instituto}\@lskipoffset{.25}%
		\@textscale{.5}{\@departamento}%
		\end{spacing}%
	\endgroup
}

\newcommand\pretextual{%
	\renewcommand{\chapteralign}{\centering}
    \renewcommand{\chapterformat}{\Huge \bfseries \thispagestyle{\style@pretextual}}
	\cleardoublepage\pagenumbering{roman}
	\@pagestyle{\style@pretextual}
	\AtBeginShipout{\stepcounter{TotPaginas}}
}

\newcommand\textual{%
	\renewcommand{\chapteralign}{\raggedright}
    \renewcommand{\chapterformat}{\Huge \bfseries \thispagestyle{\style@chapter}}
	\cleardoublepage\pagenumbering{arabic}
	\@pagestyle{\style@textual}
}

\newcommand\postextual{%
	\renewcommand{\chapteralign}{\centering}

	%lining figures in section and subsection
	\renewcommand\thesection{\thechapter.\protect\textlinf{\arabic{section}}}
	\renewcommand\thesubsection{\thesection.\protect\textlinf{\arabic{subsection}}}
	\renewcommand\thesubsubsection{\thesubsection.\protect\textlinf{\arabic{subsubsection}}}

	%avoid problem with bookmarksnumbered=true
	\pdfstringdefDisableCommands{\renewcommand*{\textlinf}[1]{##1}}
}

%work title and subtitle
\renewcommand*{\title}{\@dblarg\set@title}%
\newcommand*{\set@title}[2][]{%
	\xdef\@title{#1}
	\xdef\@maintitle{#2}
}
\newcommand*{\subtitle}{\@dblarg\set@subtitle}
\newcommand*{\set@subtitle}[2][]{%
	\gdef\@subtitle{#2}
	\xdef\@title{\expandafter{\@title}: #1}
}

%keyword definition - for cip and resumo/abstract
\newcommand*{\keywords}{\@dblarg\set@keywords}
\newcommand*{\set@keywords}[2][]{%
	\gdef\@keywords{#1}
	\gdef\@@keywords{#2}
}
\newcommand*{\palavraschave}{\@dblarg\set@palavraschave}
\newcommand*{\set@palavraschave}[2][]{%
	\gdef\@palavraschave{#1}
	\gdef\@@palavraschave{#2}
}

%work date
\def\diamesano#1#2#3{\def\@dia{#1}%
                     \def\@mes{#2}%
                     \def\@ano{#3}}

%author definition
\newcommand{\autor}[2]{
	\def\primeiroautor{#1\ #2}% Single author in order First Name
	\def\primeiroautorinv{#2,\ #1}% Single author in order Last, First
	\author{\primeiroautor}
}

%co-author definition
\newcommand{\coautor}[2]{
	\if@posgraduacao
		\ClassError{UnB-CIC-English}
			{Command \protect\coautor\space invalid for master's}
			{Only undergraduate works can have more than one author}
	\fi
	\@ifundefined{primeiroautor}{
		\ClassError{UnB-CIC-English}
			{Command \protect\autor\space must be called before \protect\coautor\space}
			{The main author must be defined before the secondary one}
	}{}
	\def\segundoautor{#1\ #2}% Secondary author in order First Name
	\def\segundoautorinv{#2,\ #1}% Secondary author in order Last, First
	\author{\primeiroautor\\ \segundoautor}
}

%course coordinator definition
\def\coordenador{\@ifnextchar[%]
              {\at@coorda}%If first optional argument calls this
              {\at@coordo}}%otherwise calls this routine

\def\at@coorda[#1]#2#3{\def\@sexocoordenador{Coordinator}%
                        \def\@nomecoordenador{#2}%
                        \def\@filicoordenador{#3}}

\def\at@coordo#1#2{\def\@sexocoordenador{Coordinator}%
                        \def\@nomecoordenador{#1}%
                        \def\@filicoordenador{#2}}

%advisor definition
\def\orientador{\@ifnextchar[%]
              {\at@orienta}%If first optional argument calls this
              {\at@oriento}}%otherwise calls this routine

\def\at@orienta[#1]#2#3{\def\@sexoorientador{Advisor}%
                        \def\@nomeorientador{#2}%
                        \def\@filiorientador{#3}}

\def\at@oriento#1#2{\def\@sexoorientador{Advisor}%
                        \def\@nomeorientador{#1}%
                        \def\@filiorientador{#2}}

%co-advisor definition
\newif\if@coorientador
\def\coorientador{\@coorientadortrue \@ifnextchar[%]
              {\at@coorienta}%If first optional argument calls this
              {\at@cooriento}}%otherwise calls this routine

\def\at@coorienta[#1]#2#3{\def\@sexocoorientador{Co-advisor}%
                        \def\@nomecoorientador{#2}%
                        \def\@filicoorientador{#3}}%
                        %%%\@defsegundomembro{#2}{#3}}

\def\at@cooriento#1#2{\def\@sexocoorientador{Co-advisor}%
                        \def\@nomecoorientador{#1}%
                        \def\@filicoorientador{#2}}%
                        %%%\@defsegundomembro{#1}{#2}}

% committee definition
% the advisor is always the first committee member
% the co-advisor, if any, is the second committee member
% the fourth member is optional
\newif\if@quartomembro
\def\@defsegundomembro#1#2{\def\@nomedois{#1}\def\@filidois{#2}}
\def\@defterceiromembro#1#2{\def\@nometres{#1}\def\@filitres{#2}}
\def\@defquartomembro#1#2{\def\@nomequatro{#1}\def\@filiquatro{#2}\@quartomembrotrue}

% If doctorate, there is a fifth member
\newif\if@quintomembro\@quintomembrofalse%
\if@doutorado%
	\def\@defquintomembro#1#2{\def\@nomecinco{#1}\def\@filicinco{#2}}%
\fi%

\def\membrobanca#1#2{
	\ifx\@nomedois\undefined
		\@defsegundomembro{#1}{#2}
	\else
		\ifx\@nometres\undefined
			\@defterceiromembro{#1}{#2}
		\else
			\ifx\@nomequatro\undefined
				\@defquartomembro{#1}{#2}
			\else
                \if@doutorado%
                    \if@quintomembro%
                        \ClassError{UnB-CIC-English}%
                            {The committee can only have 5 members}%
                            {Remember that advisor and co-advisor are already part of the committee}%
                    \else
                        \@defquintomembro{#1}{#2}%
                        \@quintomembrotrue%
                    \fi%
                \else%
                    \ClassError{UnB-CIC-English}%
                        {The committee can only have 3 or 4 members}%
                        {Remember that advisor and co-advisor are already part of the committee}%
                \fi%
			\fi
		\fi
	\fi
}

\newenvironment{paginatitulo}
	{\cleardoublepage\thispagestyle{empty}}
	{}

\newenvironment{agradecimentos}
	{\begin{paginatitulo}%
	\pdfbookmark{Acknowledgements}{acknowledgements}%
		\chapter*{Acknowledgements}}%
	{\end{paginatitulo}}

\newenvironment{disclaimer}
	{\begin{paginatitulo}%
	\pdfbookmark{Disclaimer}{disclaimer}%
		\chapter*{Disclaimer}}%
	{\end{paginatitulo}}

\renewenvironment{abstract}
	{\begin{paginatitulo}%
	\begin{otherlanguage*}{english}
	\pdfbookmark{Abstract}{abstract}%
		\chapter*{Abstract}}%
		{\vskip 15pt\noindent\textbf{Keywords:}\ \@@keywords%
	\end{otherlanguage*}
	\end{paginatitulo}}

\newenvironment{resumo}
	{\begin{paginatitulo}%
	\begin{otherlanguage*}{brazil}
	\pdfbookmark{Resumo}{resumo}%
	\chapter*{Resumo}}%
		{\vskip 15pt\noindent\textbf{Palavras-chave:}\ \@@palavraschave%
	\end{otherlanguage*}
    \end{paginatitulo}}

\newenvironment{dedicatoria}
	{\begin{paginatitulo}%
	\pdfbookmark{Dedication}{dedication}%
		\chapter*{Dedication}}%
	{\end{paginatitulo}}

\def\@printfulltitle{%
	\@ifundefined{@subtitle}{%
		\textbf{\Large \@title}%
	}{%
		\textbf{\Large\@maintitle\\[1ex]\large\@subtitle}%
	}\par%
}

\newcommand\capa{
  \begin{paginatitulo}%
	\onehalfspacing%
        \begin{center}%
             \@logo[contorno_preto]
             \null\vfill%
             \@printfulltitle%
             \vfill%
             {\large \@author}%
             \vfill%
            {\large \@cidade}\\
            {\large \@ano}%
       \end{center}%
	\defaultspacing%
  \end{paginatitulo}%
}

\newcommand\folharosto{%
	\begin{paginatitulo}%
		\onehalfspacing%
		\begin{center}%
			\@logo\vfill%
			\@printfulltitle\vfill%
			{\large\@author}\vfill%
			\@mododeescrita\ %
			\if@posgraduacao%
				\if@ppca%
					submitted as a partial requirement for %
					\if@qualificacao{qualifying for}\else{completion of}\fi~the\\%
				\else%
					submitted as a partial requirement for\\
					\if@qualificacao{qualifying for}\else{completion of}\fi~the %
				\fi%
				\@nomecurso\ in \@nomeDoPrograma%
			\else%
				submitted as a partial requirement\\
				for completion of the \@nomecurso%
			\fi%
			\vfill%
			\@sexoorientador\\\@nomeorientador\\%
			\if@coorientador
				\vskip 1cm\@sexocoorientador\\\@nomecoorientador
			\fi
			\vfill%
			{\large\@cidade}\\
			{\large\@ano}%
		\end{center}%
		\defaultspacing%
	\end{paginatitulo}%
}

\newcommand\folhaaprovacao{%
	\begin{paginatitulo}%
		\onehalfspacing%
		\begin{center}%
			\@logo\vfill%
			\@printfulltitle\vfill%
			{\large\@author}\vfill%
			\@mododeescrita\ %
			\if@posgraduacao%
				\if@ppca%
					submitted as a partial requirement for %
					\if@qualificacao{qualifying for}\else{completion of}\fi~the\\%
				\else%
					submitted as a partial requirement for\\
					\if@qualificacao{qualifying for}\else{completion of}\fi~the %
				\fi%
				\@nomecurso\ in \@nomeDoPrograma%
			\else%
				submitted as a partial requirement\\
				for completion of the \@nomecurso%
			\fi%
			\vfill%
			\@nomeorientador\ (\@sexoorientador)\\
			\@filiorientador\\[1cm]
			\if@quintomembro%
				\begin{tabular}{cc}
					\@nomedois & \@nometres\\
					\@filidois & \@filitres\\[1cm]%
					\@nomequatro & \@nomecinco\\%
					\@filiquatro & \@filicinco%
				\end{tabular}\\[1cm]
			\else%
				\begin{tabular}{cc}
					\@nomedois & \@nometres\\
					\@filidois & \@filitres%
				\end{tabular}\\[1cm]
				\if@quartomembro%
					\@nomequatro\\%
					\@filiquatro\\[1cm]
				\fi%
			\fi%
			\@nomecoordenador\\
			\@sexocoordenador\ of the\ %
			\if@posgraduacao Graduate Program in \@nomeDoPrograma\else \@nomecurso\fi%
			\vfill%
			{\large \@cidade, \@dia\ \@mes\ \@ano}
		\end{center}%
		\defaultspacing%
	\end{paginatitulo}%
}

%%
%% --- Document end and beginning processing ---
%%
\AtEndDocument{%
	\if@PosTextualNaoIncluso\@conteudoPosTextual\fi%

	\addtocounter{page}{-1}%decrement to later . . .
    \ifhyperref{%
		\immediate\write\@mainaux{%
			\string\newlabel{TotPaginas}{{\arabic{TotPaginas}}{\thepage}{}{page.\thepage}{}}%
		}%
	}{%
		\immediate\write\@mainaux{%
			\string\newlabel{TotPaginas}{{\arabic{TotPaginas}}{\thepage}}%
		}%
	}
	\addtocounter{page}{1}% . . . increment again
}

\AtBeginDocument{%
	%define PDF properties and load auxiliary packages
	\@ifpackageloaded{hyperref}{
	\hypersetup{
            pdftitle=\@title,
            pdfauthor=\@author,
            pdfsubject=\@assunto,
            pdfkeywords=\@palavraschave}
		%better position links for captions
        \RequirePackage[all]{hypcap}
	}{
		\RequirePackage{nohyperref} %dummy macros for hyperref
		\RequirePackage{url} %allows line breaks in URLs
	}

	%control table of contents according to current style,
	%after hyperref loaded because it overwrites
	%the \addcontentsline command
	\let\@addcontentsline\addcontentsline
	\def\@emptystr{empty}
	\def\@pagestyle#1{
		\let\@styletoset=#1%
		\pagestyle{\@styletoset}%
		%if style is empty, don't add to TOC
		\ifx\@styletoset\@emptystr
			\renewcommand\addcontentsline[3]{\relax}
		\else
			\let\addcontentsline\@addcontentsline
		\fi
	}

	%translations for glossaries.sty
	\@ifpackageloaded{glossaries}{%
		\addto\captionsamerican{%
		\renewcommand*{\glossaryname}{Glossary}%
		\renewcommand*{\acronymname}{List of Abbreviations and Acronyms}%
		\renewcommand*{\entryname}{Entry}%
		\renewcommand*{\descriptionname}{Description}%
		\renewcommand*{\symbolname}{Symbol}%
		\renewcommand*{\pagelistname}{Page list}%
		\renewcommand*{\glssymbolsgroupname}{Symbols}%
		\renewcommand*{\glsnumbersgroupname}{Numbers}}
	}{}

	\@ifundefined{XeTeXversion}%
	{%support for Adobe Minion Pro font - LaTeX package
		\@ifpackageloaded{MinionPro}{}{%
			\newcommand{\figureversion}[1]{\relax}
		}
		\newenvironment{linfig}{\figureversion{lining}}{\figureversion{text}}%
		\newenvironment{tabfig}{\figureversion{tabular}}{\figureversion{proportional}}%
		\newcommand*{\textlinf}[1]{\figureversion{lining}#1\figureversion{text}}%
		\newcommand*{\texttabf}[1]{\figureversion{tabular}#1\figureversion{proportional}}%
		\newcommand*{\@tablefigures}{\figureversion{lining,tabular}}%
	}%
	{%fontspec support - XeTeX
 		\def\@textlinf{\protect\addfontfeatures{Numbers=Lining}}%
 		\def\@texttabf{\protect\addfontfeatures{Numbers=Monospaced}}%
		\newenvironment{linfig}{\bgroup\@textlinf}{\egroup}%
		\newenvironment{tabfig}{\bgroup\@texttabf}{\egroup}%
		\newcommand*{\textlinf}[1]{{\@textlinf #1}}%
		\newcommand*{\texttabf}[1]{{\@texttabf #1}}%
		\newcommand*{\@tablefigures}{\addfontfeature{Numbers={Lining,Monospaced}}}%
		\newcommand*{\slashedzero}{{\addfontfeature{Numbers={Lining,SlashedZero}}0}}%
	}%

	\if@baselineskip
		%to maintain initial value
		\newdimen\@oldbaseline

		%define spacing via baselinekip
		\@oldbaseline=\baselineskip
		\newcommand{\setstretch}[1]{\setlength{\baselineskip}{#1\@oldbaseline}}
		\newcommand{\singlespacing}{\setstretch{1}}
		\newcommand{\onehalfspacing}{\setstretch{1.35}}
		\newcommand{\doublespacing}{\setstretch{1.7}}
	\fi

	\newcommand{\defaultspacing}{%
		\if@singlespacing\singlespacing%single spacing
			\else\if@onehalfspacing\onehalfspacing%one and half spacing
				\else\doublespacing%double spacing
			\fi%
		\fi}
	\defaultspacing% initial call

	% Committee member control
	\ifx\@nometres\undefined%
        \ClassError{UnB-CIC-English}%
            {The committee must have at least 3 members}%
            {Remember that advisor and co-advisor are already part of the committee}%
    \fi%
    \if@doutorado%
		\if@qualificacao%
		\else%
	        \if@quintomembro%
	        \else%
	            \ClassError{UnB-CIC-English}%
	                {The committee must have 5 members}%
	                {Remember that advisor and co-advisor are already part of the committee}%%
	        \fi%
	    \fi%
    \fi%

	\@conteudoPreTextual%
}


%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% New Commands %%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%
\newcommand{\titulo}[1]{\title{#1}}%

\def\@texpath{tex}%
\newcommand{\@erroSeNaoExiste}[1]{%
    \IfFileExists{#1}{}%
	    {\ClassError{UnB-CIC-English}%
	        {File "#1" not found.}
	        {The file "#1" is required.}}%
}%

\newcommand{\@inputTeXFile}[1]{%
	\@erroSeNaoExiste{\@texpath/#1}%
	\input{\@texpath/#1}%
}%
\newcommand{\@IncluirSecao}[1]{%
	\begin{#1}%
		\@inputTeXFile{#1}%
	\end{#1}%
}%

\newcommand{\capitulo}[2]{%
	\chapter{#2}%
	\label{#1}%
	\@inputTeXFile{#1}%
}%

\newif\if@PosTextualNaoIncluso%
\@PosTextualNaoInclusotrue%

\newcounter{apendice}\setcounter{apendice}{0}%
\newcommand{\apendice}[2]{%
	\addtocounter{apendice}{1}%
	\ifnum\value{apendice}<2%
		\if@PosTextualNaoIncluso\@conteudoPosTextual\fi%
	    \appendix%
	    \addcontentsline{toc}{chapter}{\appendixname}%
	\fi%
	\capitulo{#1}{#2}%
}%

\newcounter{anexo}\setcounter{anexo}{0}%
\newcommand{\anexo}[2]{%
	\addtocounter{anexo}{1}%
	\ifnum\value{anexo}<2%
		\if@PosTextualNaoIncluso\@conteudoPosTextual\fi%
		\renewcommand{\appendixname}{Annex}%
		\ifnum\value{apendice}<1\appendix\fi%
		\renewcommand{\thechapter}{\Roman{anexo}}%
	    \addcontentsline{toc}{chapter}{\appendixname}%
	\fi%
	\capitulo{#1}{#2}%
}%

\newcommand{\@listofacronyms}{%
	\@inputTeXFile{siglas}%
	\printglossary[type=\acronymtype]%
}%

\newcommand{\@conteudoPreTextual}{%
	\maketitle%
	\if@qualificacao\else%
		\@IncluirSecao{dedicatoria}%
		\@IncluirSecao{agradecimentos}%
		\@IncluirSecao{disclaimer}%

		% *DO NOT REMOVE*, it is important to acknowledge this CAPES support.
		This work was carried out with support from the Coordination for the Improvement of Higher Education Personnel - Brazil (CAPES), through access to the Portal of Journals.
	\fi%
	\@IncluirSecao{resumo}%
	\selectlanguage{american}%
	\@IncluirSecao{abstract}%
	\selectlanguage{american}%
    \selectbiblanguage{american}%
	\@erroSeNaoExiste{bibliografia.bib}%

	\tableofcontents%
	\iftotalfigures\listoffigures\fi%
	\iftotaltables\listoftables\fi%
	\@listofacronyms%

	\textual%
}%

\newcommand{\@conteudoPosTextual}{%
	\@PosTextualNaoInclusofalse%
    \postextual%
	\bibliographystyle{babunsrt}%
    \bibliography{bibliografia}%
}%


\newcommand{\sigla}[2]{\newacronym{#1}{#1}{#2}}%

\newcommand{\@referencia}[2]{#1~\ref{#2}}%
\newcommand{\refCap}[1]{\@referencia{Chapter}{#1}}%
\newcommand{\refCaps}[2]{\@referencia{Chapters}{#1}\ to \ref{#2}}%
\newcommand{\refFig}[1]{\@referencia{Figure}{#1}}%
\newcommand{\refFigs}[2]{\@referencia{Figures}{#1}\ to \ref{#2}}%
\newcommand{\refEq}[1]{\@referencia{Equation}{#1}}%
\newcommand{\refEqs}[2]{\@referencia{Equations}{#1}\ to \ref{#2}}%
\newcommand{\refTab}[1]{\@referencia{Table}{#1}}%
\newcommand{\refTabs}[2]{\@referencia{Tables}{#1}\ to \ref{#2}}%
\newcommand{\refApendice}[1]{\@referencia{Appendix}{#1}}%
\newcommand{\refAnexo}[1]{\@referencia{Annex}{#1}}%

\newcommand{\figuraBib}[6][!ht]{%
	\begin{figure}[#1]%
		\begin{center}%
			\includegraphics[#6]{#2}%
			\caption[#3.]{#3\ifx#4\empty\else~(Source: ~\cite{#4})\fi.\label{#5}}%
		\end{center}%
	\end{figure}%
}%
\newcommand{\figura}[5][!ht]{%
	\figuraBib[#1]{#2}{#3}{}{#4}{#5}%
}%

\newcommand{\equacao}[2]{%
	\begin{equation}%
		\label{#1}%
		\begin{gathered}%
			#2
		\end{gathered}%
	\end{equation}%
}%

\newcommand{\tabelaBib}[5]{%
	\begin{table}%
		\centering%
		\caption[#1.]{#1\ifx#2\empty\else~(Source: ~\cite{#2})\fi.\label{#3}}%
		\begin{tabular}{#4}%
			#5
		\end{tabular}%
	\end{table}%
}%

\newcommand{\tabela}[4]{%
	\tabelaBib{#1}{}{#2}{#3}{#4}%
}%


\endinput

