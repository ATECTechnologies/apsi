;;; apsi-mode.el --- Major mode for the Apsi language  -*- lexical-binding: t; -*-

;; Load it from your init file:   (load "/path/to/apsi/editors/emacs/apsi-mode.el")
;; Indentation comes from python-mode (Apsi uses Python-style blocks); the language server
;; starts with M-x eglot (registered below).

;;; Code:
(require 'python)

(defconst apsi--keywords
  '("let" "var" "fn" "return" "if" "elif" "else" "while" "for" "in" "break" "continue" "struct"
    "and" "or" "not" "use" "import" "assert" "evolve" "as" "pass" "test" "parallel"
    "over" "step" "using" "tol" "record" "every" "into"))

(defconst apsi--types
  '("Float" "Int" "Bool" "Str" "List" "Tuple" "Dict" "Any" "Fn" "Record" "Tensor" "Vec" "Vec2" "Vec3" "Vec4"
    "Dimensionless" "Angle" "Length" "Mass" "Time" "Current" "Temperature" "Amount" "Area" "Volume"
    "Velocity" "Speed" "Acceleration" "Force" "Energy" "Power" "Pressure" "Momentum" "Frequency"
    "Charge" "Voltage" "Resistance" "Density" "Entropy" "MagneticField" "ElectricField"))

(defconst apsi-font-lock-keywords
  `((,(regexp-opt apsi--keywords 'symbols) . font-lock-keyword-face)
    (,(regexp-opt '("true" "false" "none" "pi" "tau" "inf" "nan") 'symbols) . font-lock-constant-face)
    (,(regexp-opt apsi--types 'symbols) . font-lock-type-face)
    ("\\_<fn\\s-+\\(\\(?:\\sw\\|\\s_\\)+\\)" 1 font-lock-function-name-face)
    ("\\_<struct\\s-+\\(\\(?:\\sw\\|\\s_\\)+\\)" 1 font-lock-type-face)
    ("^\\s-*@\\(?:\\sw\\|\\s_\\|\\.\\)+" . font-lock-preprocessor-face)
    ("\\_<[0-9][0-9_]*\\(?:\\.[0-9_]+\\)?\\(?:[eE][-+]?[0-9]+\\)?\\(\\[[^]\n]*\\]\\|[[:alpha:]µΩÅ]+\\)" 1 font-lock-builtin-face)
    ("\\_<as\\s-*\\(\\[[^]\n]*\\]\\)" 1 font-lock-builtin-face)
    ("±\\|+/-\\||>\\|??\\|\\?\\." . font-lock-warning-face)))

;;;###autoload
(define-derived-mode apsi-mode python-mode "Apsi"
  "Major mode for editing Apsi (.apsi) files."
  (setq-local font-lock-defaults '(apsi-font-lock-keywords nil nil))
  (setq-local comment-start "# ")
  (setq-local tab-width 4)
  (setq-local python-indent-offset 4)
  (setq-local compile-command (concat "apsi run " (shell-quote-argument (or buffer-file-name "")))))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.apsi\\'" . apsi-mode))
(add-to-list 'interpreter-mode-alist '("apsi" . apsi-mode))

(with-eval-after-load 'eglot
  (add-to-list 'eglot-server-programs '(apsi-mode . ("apsi" "lsp"))))

(provide 'apsi-mode)
;;; apsi-mode.el ends here
