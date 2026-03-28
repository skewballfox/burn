use proc_macro::TokenStream;
use quote::quote;
use syn::{ImplItem, ItemImpl};

pub(crate) fn parse_backend_impl(mut input_impl: ItemImpl) -> TokenStream {
    // 1. Collect names of methods the user ALREADY implemented
    let mut implemented_methods = std::collections::HashSet::new();
    for item in &input_impl.items {
        if let ImplItem::Fn(method) = item {
            implemented_methods.insert(method.sig.ident.to_string());
        }
    }

    
    let mut extra_items = Vec::new();

    
    //boilerplate:
    if !implemented_methods.contains("seed") {
        extra_items.push(quote! {
            fn seed(device: &Self::Device, seed: u64) {
                B::seed(device, seed)
            }
        });
    }

    
    if !implemented_methods.contains("float_from_data") {
        extra_items.push(quote! {
            fn float_from_data(data: TensorData, device: &Device<Self>) -> FloatTensor<Self> {
                B::float_from_data(data, device)
            }
        });
    }

    // 3. Append the generated methods to the impl block
    for item in extra_items {
        input_impl.items.push(syn::parse2(item).unwrap());
    }

    TokenStream::from(quote! { #input_impl })
}
